import pandas as pd
import numpy as np
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
import logging
import json
import os
from tqdm import tqdm
import gc
import argparse
from peft import LoraConfig, get_peft_model, TaskType
import random

# --- Environment and Logging Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# --- Data Loading ---
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
    test_df = load_multiple_datasets(test_paths) if test_paths else pd.DataFrame() # Handle case with no test data
    
    return train_df, dev_df, test_df

# --- Prompt Engineering ---
SYSTEM_PROMPT = """You are a discourse relation classifier. Your task is to analyze text pairs and classify their discourse relationship and label them from the given labels.

IMPORTANT: Your response must be ONLY a JSON object with the format {"label": "your_classification"}
Do not include any other text or explanations outside of the JSON."""


# --- Dataset Class ---
class DiscourseClassificationDataset(Dataset):
    """Efficient instruction fine-tuning dataset that pre-processes static prompts."""
    def __init__(self, df: pd.DataFrame, processor, all_labels: list[str], max_length=2048):
        self.df = df
        self.processor = processor
        self.max_length = max_length
        self.all_labels_str = ", ".join([f"'{label}'" for label in all_labels])

        # Pre-build the static part of the prompt (system + few-shot)
        # static_messages = [{"role": "system", "content": SYSTEM_PROMPT}] 
        # self.static_prompt_text = self.processor.apply_chat_template(
        #     static_messages, tokenize=False, add_generation_prompt=False
        # )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dir_explanation = "Unit 1's statement points to Unit 2." if row['dir'] == '1>2' else "Unit 2's statement points to Unit 1."

        # Create the dynamic part of the prompt (the current example)
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
        
        # Combine static prompt with the dynamic user/assistant turn
        dynamic_turn_text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json_label}
            ],
            tokenize=False, add_generation_prompt=False
        )

        # Remove BOS token from the dynamic part to avoid duplication
        if dynamic_turn_text.startswith(self.processor.bos_token):
            dynamic_turn_text = dynamic_turn_text[len(self.processor.bos_token):]

        full_text = dynamic_turn_text #self.static_prompt_text + dynamic_turn_text

        # Tokenize the final combined text
        inputs = self.processor(
            text=full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        tokenized_inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        tokenized_inputs['length'] = len(tokenized_inputs['input_ids'])
        return tokenized_inputs

# --- Evaluation and Prediction ---
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
            outputs = model.generate(**inputs, max_new_tokens=20, eos_token_id=processor.eos_token_id,  pad_token_id=processor.pad_token_id)
        
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
    """Custom trainer to handle generative evaluation."""
    def __init__(self, *args, processor=None, eval_df=None, all_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_df = eval_df
        self.all_labels = all_labels
        self.processor = processor
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Create prompts for the evaluation set
        prompts = []
        all_labels_str =  ", ".join([f"'{label}'" for label in self.all_labels])

        for _, row in self.eval_df.iterrows():
            # This is the same logic as the dataset, but without the answer
            dir_explanation = "Unit 1's statement points to Unit 2." if row['dir'] == '1>2' else "Unit 2's statement points to Unit 1."
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

            messages = [{"role": "user", "content": user_prompt}] #[{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": "user", "content": user_prompt}]
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)

        # Generate and parse predictions
        predictions = generate_predictions(self.model, self.processor, prompts, self.args.per_device_eval_batch_size, self.all_labels)
        true_labels = self.eval_df['label'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted', labels=self.all_labels, zero_division=0)
        
       # Calculate per-class metrics for debugging
        from sklearn.metrics import classification_report
        report = classification_report(true_labels, predictions, labels=self.all_labels, zero_division=0, output_dict=True)
        
        # Log detailed metrics
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted F1: {f1:.4f}")
        logger.info(f"Failed predictions (n/a): {predictions.count('n/a')}/{len(predictions)}")
        logger.info(f"{'='*50}\n")
        
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_f1": f1,
            f"{metric_key_prefix}_loss": 0.0  # Dummy loss to satisfy trainer
        }
        
        self.log(metrics)
        return metrics

# --- Main Execution ---
def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # --- GPU Optimizations ---
    if torch.cuda.is_available():
        logger.info("Applying GPU optimizations...")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        # Note: torch.backends.cudnn.benchmark is set to False in set_seed() for reproducibility

    # 1. Load and Clean Data
    try:
        train_df, dev_df, test_df = load_data_from_directory(args.data_dir)
        for df in [train_df, dev_df, test_df]:
            df['label'] = df['label'].astype(str).str.strip()
            df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
            df = df[df['label'] != '']
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Fatal error loading data: {e}")
        return

    all_labels = sorted(list(set(train_df['label'].tolist())))
    logger.info(f"Loaded {len(train_df)} train, {len(dev_df)} dev samples. Found {len(all_labels)} labels.")
    
    # 2. Setup Model, Processor, and LoRA
    logger.info(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    if processor.pad_token is None:
        processor.pad_token = processor.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation= "flash_attention_2" if torch.cuda.is_available() else "eager",
        token=True, # Or your Hugging Face token
    )
 

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Create Datasets
    train_dataset = DiscourseClassificationDataset(train_df, processor, all_labels)
    val_dataset = DiscourseClassificationDataset(dev_df, processor, all_labels)
    
    data_collator = DataCollatorForLanguageModeling(processor, mlm=False)
    
    # 4. Configure Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, 
        lr_scheduler_type="cosine", 
        warmup_ratio=0.05,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps", 
        logging_steps=100,
        eval_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=2, 
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
        label_names = [], #important for generative models
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

    # 5.. Train
    logger.info("Starting model training...")
    trainer.train()
    logger.info("Training complete.")

    # 6. Save final model
    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    logger.info(f"Model saved to {final_save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for discourse classification.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it", help="Model name from Hugging Face.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing lang subfolders with .rels files.")
    parser.add_argument("--output_dir", type=str, default="./discourse_gemma_output", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size per device.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    cli_args = parser.parse_args()
    main(cli_args)