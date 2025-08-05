import pandas as pd
import numpy as np
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset
import logging
import json
import os
from tqdm import tqdm
import gc
import argparse
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import random

# --- Environment and Logging Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Assumes preprocess.py with rel_reader1 is available
from preprocess import rel_reader1

# --- Reproducibility Setup ---
def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")

# --- Data Loading & Preparation ---

def create_sample_id(df: pd.DataFrame) -> pd.Series:
    """Creates a unique ID for each sample to ensure correct merging."""
    return df['doc'].astype(str) + "_" + df['unit1_toks'].astype(str) + "_" + df['unit2_toks'].astype(str) + "_" + df['label'].astype(str) 

def load_multiple_datasets(dataset_paths: dict[str, str]) -> pd.DataFrame:
    """Loads and combines multiple datasets from different languages."""
    combined_dfs = []
    for lang_code, file_path in dataset_paths.items():
        logger.info(f"Loading {lang_code} dataset from {file_path}")
        try:
            content = rel_reader1(file_path)
            if not content: continue
            df = pd.DataFrame(content, columns=['doc', 'unit1_toks', 'unit1_txt', 'unit2_toks', 'unit2_txt', 'unit1_snt', 'unit2_snt', 'dir', 'label'])
            df['language'] = lang_code
            combined_dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing {lang_code} dataset at {file_path}: {e}")
            continue
    if not combined_dfs:
        raise ValueError("No datasets were successfully loaded. Check paths and data integrity.")
    return pd.concat(combined_dfs, ignore_index=True)

def load_data_from_directory(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all train/dev/test datasets from a directory."""
    train_paths, dev_paths, test_paths = {}, {}, {}
    for lang_folder in os.listdir(data_dir):
        lang_path = os.path.join(data_dir, lang_folder)
        if os.path.isdir(lang_path):
            for split in ["train", "dev", "test"]:
                file_path = os.path.join(lang_path, f"{lang_folder}_{split}.rels")
                if os.path.exists(file_path):
                    if split == "train": train_paths[lang_folder] = file_path
                    elif split == "dev": dev_paths[lang_folder] = file_path
                    else: test_paths[lang_folder] = file_path

    if not train_paths or not dev_paths:
        raise FileNotFoundError("Could not find required train and dev .rels files in subdirectories.")
    
    train_df = load_multiple_datasets(train_paths)
    dev_df = load_multiple_datasets(dev_paths)
    test_df = load_multiple_datasets(test_paths) if test_paths else pd.DataFrame()
    
    return train_df, dev_df, test_df

# --- Prompt Engineering ---
SYSTEM_PROMPT = """You are a discourse relation classifier. Your task is to analyze text pairs and classify their discourse relationship and label them from the given labels.

IMPORTANT: Your response must be ONLY a JSON object with the format {"label": "your_classification"}
Do not include any other text or explanations outside of the JSON."""

# --- Dataset Class (Context-Aware) ---
class DiscourseClassificationDataset(Dataset):
    """
    Efficient instruction fine-tuning dataset that handles two types of prompts:
    1. Standard prompts for regular training data (replay samples).
    2. Rationale-guided prompts for samples with expert reasoning.
    """
    def __init__(self, df: pd.DataFrame, processor, all_labels: list[str], max_length=2048):
        self.df = df.reset_index(drop=True)  # Reset index to avoid issues
        self.processor = processor
        self.max_length = max_length
        self.all_labels_str = ", ".join([f"'{label}'" for label in all_labels])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dir_explanation = "Unit 1's statement points to Unit 2." if row['dir'] == '1>2' else "Unit 2's statement points to Unit 1."
        user_prompt = ""

        # Check if a rationale is present and use the appropriate prompt
        if 'rationale' in self.df.columns and pd.notna(row.get('rationale')):
            # --- RATIONALE PROMPT ---
            user_prompt = f"""{SYSTEM_PROMPT}
Analyze the discourse relation between **Unit 1** and **Unit 2**. 
**Available Labels**: [{self.all_labels_str}]
**Unit 1:** "{row['unit1_txt']}"
**Unit 2:** "{row['unit2_txt']}"
**Direction:** {row['dir']} ({dir_explanation})

**Expert Analysis:** "{row['rationale']}"

Full Context:
- Unit 1: "{row['unit1_snt']}"
- Unit 2: "{row['unit2_snt']}"

Respond with ONLY a JSON object: {{"label": "your_classification"}}"""
        else:
            # --- STANDARD PROMPT (for replay samples) ---
            user_prompt = f"""{SYSTEM_PROMPT}
Analyze the discourse relation between **Unit 1** and **Unit 2**. 
**Available Labels**: [{self.all_labels_str}]
**Unit 1:** "{row['unit1_txt']}"
**Unit 2:** "{row['unit2_txt']}"
**Direction:** {row['dir']} ({dir_explanation})

Full Context:
- Unit 1: "{row['unit1_snt']}"
- Unit 2: "{row['unit2_snt']}"

Respond with ONLY a JSON object: {{"label": "your_classification"}}"""

        json_label = json.dumps({"label": row['label']})
        
        full_text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json_label}
            ],
            tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        tokenized_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        tokenized_inputs['length'] = len(tokenized_inputs['input_ids'])
        if 'rationale' in self.df.columns and pd.notna(row.get('rationale')):
            tokenized_inputs['sample_weight'] = 1.5  # Increased from 1.2
        else:
            tokenized_inputs['sample_weight'] = 1.0
        return tokenized_inputs

# --- Custom Data Collator to handle sample weights ---
class WeightedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Extract sample weights before calling parent
        sample_weights = torch.tensor([f.pop('sample_weight', 1.0) for f in features])
        
        # Call parent collator
        batch = super().__call__(features)
        
        # Add sample weights back
        batch['sample_weight'] = sample_weights
        
        return batch

# --- Evaluation Functions ---
def generate_predictions(model, processor, prompts: list[str], batch_size: int, all_labels: list[str]) -> list[str]:
    """Generates and robustly parses JSON predictions for a list of prompts."""
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Predictions"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = processor(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                eos_token_id=processor.eos_token_id,  
                pad_token_id=processor.pad_token_id,
                temperature=0.1,  # Lower temperature for more deterministic outputs
                do_sample=False   # Greedy decoding for consistency
            )
        
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

        for output in decoded_outputs:
            pred = "n/a"
            output = output.strip()
            
            # Try to parse JSON first
            try:
                # Find JSON object
                start_idx = output.find('{')
                end_idx = output.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = output[start_idx:end_idx+1]
                    json_output = json.loads(json_str)
                    pred = json_output.get("label", "n/a")
            except (json.JSONDecodeError, IndexError):
                # Fallback: try to find any valid label in the output
                output_lower = output.lower()
                for label in all_labels:
                    if label.lower() in output_lower:
                        pred = label
                        break
                
                if pred == "n/a":
                    logger.warning(f"Failed to extract label from output: {output[:100]}...")
            
            predictions.append(pred)
            
    return predictions

class DiscourseTrainer(Trainer):
    def __init__(self, *args, processor=None, eval_df=None, all_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_df = eval_df
        self.all_labels = all_labels
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to use sample weights."""
        # Extract sample weights
        sample_weights = inputs.pop("sample_weight", None)
        
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        # Calculate per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Apply sample weights if provided
        if sample_weights is not None:
            batch_size = inputs["labels"].size(0)
            seq_length = shift_labels.size(1)
            
            # Reshape loss to [batch_size, seq_length]
            loss = loss.view(batch_size, seq_length)
            
            # Create mask for non-padding tokens
            mask = (shift_labels != -100).float()
            
            # Apply weights per sample
            weighted_loss = loss * sample_weights.unsqueeze(1)
            
            # Compute mean over non-padding tokens, then weighted mean over batch
            loss = (weighted_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # This evaluation logic remains unchanged, always testing the model's unaided performance.
        prompts = []
        all_labels_str = ", ".join([f"'{label}'" for label in self.all_labels])
        
        for _, row in self.eval_df.iterrows():
            dir_explanation = "Unit 1's statement points to Unit 2." if row['dir'] == '1>2' else "Unit 2's statement points to Unit 1."
            # Use standard prompt WITHOUT rationale for evaluation
            user_prompt = f"""{SYSTEM_PROMPT}
Analyze the discourse relation between **Unit 1** and **Unit 2**.
**Available Labels**: [{all_labels_str}]
**Unit 1:** "{row['unit1_txt']}"
**Unit 2:** "{row['unit2_txt']}"
**Direction:** {row['dir']} ({dir_explanation})

Full Context:
- Unit 1: "{row['unit1_snt']}"
- Unit 2: "{row['unit2_snt']}"

Respond with ONLY a JSON object: {{"label": "your_classification"}}"""
            messages = [{"role": "user", "content": user_prompt}]
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)

        # Generate and parse predictions
        predictions = generate_predictions(
            self.model, 
            self.processor, 
            prompts, 
            self.args.per_device_eval_batch_size, 
            self.all_labels
        )
        true_labels = self.eval_df['label'].tolist()
        
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted', labels=self.all_labels, zero_division=0)
        
        # Calculate per-class metrics for debugging
        report = classification_report(true_labels, predictions, labels=self.all_labels, zero_division=0, output_dict=True)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {f1:.4f}")
        logger.info(f"Failed predictions (n/a): {predictions.count('n/a')}/{len(predictions)}")
        logger.info(f"{'='*50}\n")
        
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_f1": f1,
            f"{metric_key_prefix}_loss": 0.0  # Dummy loss for generative models
        }
        
        self.log(metrics)
        return metrics

# --- Main Execution (NEW Data Curation Logic) ---
def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # --- GPU Optimizations ---
    if torch.cuda.is_available():
        logger.info("Applying GPU optimizations...")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        # Note: torch.backends.cudnn.benchmark is set to False in set_seed() for reproducibility

    # 1. Load All Necessary Data
    try:
        original_train_df, dev_df, test_df = load_data_from_directory(args.data_dir)
        
        # Clean data
        for df in [original_train_df, dev_df, test_df]:
            if not df.empty:
                df['label'] = df['label'].astype(str).str.strip()
                df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
                df = df[df['label'] != '']
        
        rationale_df = pd.read_json(args.rationale_path, lines=True) if args.rationale_path else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Fatal error during initial data loading: {e}")
        return

    # 2. Prepare the Curriculum Learning Dataset
    if not rationale_df.empty:
        logger.info("--- Starting Data Preparation  ---")

        # Create unique IDs to identify the hard samples
        original_train_df['sample_id'] = create_sample_id(original_train_df)
        rationale_df['sample_id'] = create_sample_id(rationale_df)
        hard_sample_ids = set(rationale_df['sample_id'])

        # Create the pool of correctly predicted samples by excluding the hard samples
        correct_samples_pool_df = original_train_df[~original_train_df['sample_id'].isin(hard_sample_ids)].copy()
        logger.info(f"Full training set: {len(original_train_df)}, Hard samples: {len(hard_sample_ids)}")
        logger.info(f"Created pool of correctly predicted samples: {len(correct_samples_pool_df)}")
        
        # Determine number of replay samples from the "correct" pool
        # num_rationale_samples = len(rationale_df)
        # num_replay_samples = int(num_rationale_samples * args.replay_ratio)
        
        # if len(correct_samples_pool_df) < num_replay_samples:
        #     logger.warning(f"Requested {num_replay_samples} replay samples, but only {len(correct_samples_pool_df)} are available. Using all available.")
        #     num_replay_samples = len(correct_samples_pool_df)

        # logger.info(f"Hard samples (with rationales): {num_rationale_samples}")
        # logger.info(f"Easy samples for replay (from correct pool): {num_replay_samples} (Ratio: {args.replay_ratio})")

        # # Create the replay set by sampling from the "correct" pool
        # if num_replay_samples > 0:
        #     replay_df = correct_samples_pool_df.sample(n=num_replay_samples, random_state=42)
        # else:
        #     replay_df = pd.DataFrame()
        replay_df = correct_samples_pool_df.copy()
        logger.info(f"Using ALL {len(replay_df)} correctly predicted samples (no sampling)")
        # Combine the hard set and the replay set
        train_df = pd.concat([rationale_df, replay_df], ignore_index=True)
        
        # Shuffle the combined dataset with the provided seed
        train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        
        logger.info(f"Final curated training set size: {len(train_df)} samples.")
    else:
        logger.info("Rationale path not provided. Using standard training set.")
        train_df = original_train_df
        
    # 3. Standard Setup (Model, Processor, etc.)
    all_labels = sorted(list(set(train_df['label'].tolist()) | set(dev_df['label'].tolist())))
    
    logger.info(f"Loading model from Stage 1: {args.stage1_model_path}")
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(args.stage1_model_path)
    if processor.pad_token is None: 
        processor.pad_token = processor.eos_token
    
    # Load the base model and apply the saved LoRA weights
    base_model_name = args.model_name  # The original base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Load the LoRA weights from Stage 1
    model = PeftModel.from_pretrained(model, args.stage1_model_path)
    
    # Create new LoRA config for Stage 2 (can use same or different parameters)
    lora_config = LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.CAUSAL_LM
    )
    
    # Important: merge the Stage 1 LoRA weights and then add new LoRA for Stage 2
    model = model.merge_and_unload()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optional: Analyze prompt lengths if requested
    if args.analyze_lengths_only:
        logger.info("--- Analyzing Prompt Lengths ---")
        analysis_dataset = DiscourseClassificationDataset(train_df, processor, all_labels, max_length=16384) 
        
        prompt_lengths = []
        num_samples_to_check = min(len(analysis_dataset), 5000)
        # Use numpy random with seed for reproducible sampling
        np.random.seed(args.seed)
        indices_to_check = np.random.choice(len(analysis_dataset), num_samples_to_check, replace=False)

        for i in tqdm(indices_to_check, desc="Analyzing prompt lengths"):
            sample = analysis_dataset[i]
            prompt_lengths.append(sample['length'])

        if prompt_lengths:
            stats_df = pd.Series(prompt_lengths).describe(percentiles=[0.90, 0.95, 0.99])
            logger.info(f"\n\n--- Prompt Length Analysis (on {num_samples_to_check} samples) ---\n{stats_df.round(2)}\n")
            
            max_observed_len = int(stats_df['max'])
            percentile_99 = int(stats_df['99%'])
            
            suggested_length = int(np.ceil(percentile_99 / 64)) * 64
            
            logger.info(f"Max observed prompt length: {max_observed_len} tokens.")
            logger.info(f"99% of prompts are shorter than {percentile_99} tokens.")
            logger.info(f"RECOMMENDATION: Based on this, a --max_length of {suggested_length} would prevent truncating over 99% of your samples.")
            
            if max_observed_len > args.max_length:
                logger.warning(f"WARNING: Your current --max_length is {args.max_length}, but the longest observed prompt is {max_observed_len}. Some training data WILL be truncated.")
            else:
                logger.info(f"Your current --max_length of {args.max_length} is sufficient.")
            logger.info("-" * 50)
        
        logger.info("Exiting after prompt length analysis as requested.")
        return

    train_dataset = DiscourseClassificationDataset(train_df, processor, all_labels, args.max_length)
    val_dataset = DiscourseClassificationDataset(dev_df, processor, all_labels, args.max_length)
    
    # Use custom data collator that preserves sample weights
    data_collator = WeightedDataCollatorForLanguageModeling(processor, mlm=False)
    
    # 4. Configure Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Reduced from 0.1 for faster learning
        logging_strategy="steps", 
        logging_steps=50,
        eval_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1', 
        greater_is_better=True,
        bf16=True, 
        bf16_full_eval=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
        group_by_length=True, 
        length_column_name="length",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=[],  # Important for generative models
        seed=args.seed,  # Set seed in training arguments
        data_seed=args.seed  # Set data seed for data loader sampling
    )

    trainer = DiscourseTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        data_collator=data_collator, 
        processor=processor,
        eval_df=dev_df, 
        all_labels=all_labels
    )

    logger.info("Starting Stage 2 Curated Training with Sample Weights...")
    trainer.train()
    logger.info("Training complete.")

    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    logger.info(f"Stage 2 model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Curated Fine-tuning for Discourse Classification.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it", help="Base model name from Hugging Face.")
    parser.add_argument("--stage1_model_path", type=str, required=True, help="Path to the Stage 1 fine-tuned model.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing lang subfolders with .rels files.")
    parser.add_argument("--rationale_path", type=str, required=True, help="Path to the JSONL file with failed samples and rationales.")
    parser.add_argument("--output_dir", type=str, default="./discourse_gemma_stage2_exp2", help="Output directory for Stage 2 model.")
    
    parser.add_argument("--replay_ratio", type=float, default=1.0, help="Ratio of 'easy' replay samples to 'hard' rationale samples. E.g., 1.0 for a 1:1 mix.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs for Stage 2.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Higher learning rate for Stage 2.")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    parser.add_argument("--analyze_lengths_only", action="store_true", help="Run only the prompt length analysis and then exit.")
    
    cli_args = parser.parse_args()
    main(cli_args)