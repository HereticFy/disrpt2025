import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging
import json
from tqdm import tqdm
import os

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


torch.set_float32_matmul_precision('high')
# --- Make sure your data loading functions are accessible ---
# This assumes they are in a file named preprocess.py in the same directory
# --- Restored Data Loading Functions ---
from preprocess import rel_reader1
def load_multiple_datasets(dataset_paths: dict[str, str]) -> pd.DataFrame:
    """
    Load multiple datasets from different languages and combine them.
    Relies on the user-provided `rel_reader` function.
    """
    combined_dfs = []
    
    for lang_code, file_path in dataset_paths.items():
        logger.info(f"Loading {lang_code} dataset from {file_path}")
        try:
            content = rel_reader1(file_path)
            df = pd.DataFrame(content, columns=['doc', 'unit1_toks', 'unit1_txt', 'unit2_toks', 'unit2_txt', 'unit1_snt', 'unit2_snt', 'dir', 'label'])
            df['language'] = lang_code
            combined_dfs.append(df)
            logger.info(f"  - Loaded {len(df)} samples for {lang_code}")
        
        except Exception as e:
            logger.error(f"Error loading {lang_code} dataset at {file_path}: {e}")
            continue
    
    if not combined_dfs:
        raise ValueError("No datasets were successfully loaded. Check file paths and the rel_reader function.")
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    logger.info(f"Combined dataset created with {len(combined_df)} total samples.")
    return combined_df

def load_datasets_from_directory(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all train/dev datasets from a directory automatically based on file naming conventions.
    """
    train_file_paths, dev_file_paths, test_file_paths = [], [], []
    
    for lang_folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, lang_folder)):
            train_rels_file = os.path.join(data_dir, lang_folder, f"{lang_folder}_train.rels")
            if os.path.exists(train_rels_file):
                train_file_paths.append(train_rels_file)
            
            dev_rels_file = os.path.join(data_dir, lang_folder, f"{lang_folder}_dev.rels")
            if os.path.exists(dev_rels_file):
                dev_file_paths.append(dev_rels_file)

            test_rels_file = os.path.join(data_dir, lang_folder, f"{lang_folder}_test.rels")
            if os.path.exists(test_rels_file):
                test_file_paths.append(test_rels_file)

    if not train_file_paths:
        raise ValueError(f"No training files (*_train.rels) found in subdirectories of {data_dir}")
    if not dev_file_paths:
        raise ValueError(f"No development files (*_dev.rels) found in subdirectories of {data_dir}")
    if not test_file_paths:
        raise ValueError(f"No test files (*_test.rels) found in subdirectories of {data_dir}")

    train_dataset_paths, dev_dataset_paths, test_dataset_paths = {}, {}, {}
    for file_path in train_file_paths:
        lang_code = os.path.basename(file_path).split('_')[0]
        train_dataset_paths[lang_code] = file_path
    
    for file_path in dev_file_paths:
        lang_code = os.path.basename(file_path).split('_')[0]
        dev_dataset_paths[lang_code] = file_path

    for file_path in test_file_paths:
        lang_code = os.path.basename(file_path).split('_')[0]
        test_dataset_paths[lang_code] = file_path
    
    logger.info(f"Found training datasets for languages: {list(train_dataset_paths.keys())}")
    logger.info(f"Found development datasets for languages: {list(dev_dataset_paths.keys())}")
    logger.info(f"Found test datasets for languages: {list(test_dataset_paths.keys())}")
    
    train_df = load_multiple_datasets(train_dataset_paths)
    dev_df = load_multiple_datasets(dev_dataset_paths)
    test_df = load_multiple_datasets(test_dataset_paths)
    return train_df, dev_df, test_df


SYSTEM_PROMPT = """You are a discourse relation classifier. Your task is to analyze text pairs and classify their discourse relationship and label them from the given labels.

IMPORTANT: Your response must be ONLY a JSON object with the format {"label": "your_classification"}
Do not include any other text or explanations outside of the JSON."""

def generate_predictions(model, processor, prompts: list[str], batch_size: int, all_labels: list[str]) -> list[str]:
    """Generates and robustly parses JSON predictions for a list of prompts."""
    # This function is identical to your original script
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Predictions on Training Data"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = processor(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, eos_token_id=processor.eos_token_id, pad_token_id=processor.pad_token_id, disable_compile=True)
        
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

        for output in decoded_outputs:
            pred = "n/a"
            try:
                start_idx = output.find('{')
                end_idx = output.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = output[start_idx:end_idx+1]
                    json_output = json.loads(json_str)
                    pred = json_output.get("label", "n/a")
            except (json.JSONDecodeError, IndexError):
                output_lower = output.lower()
                for label in all_labels:
                    if label.lower() in output_lower:
                        pred = label
                        break
            predictions.append(pred)
    return predictions

def main(args):
    # 1. Load Model and Processor (Same as your script)
    logger.info(f"Loading base model: {args.base_model_name}")
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    if processor.pad_token is None: processor.pad_token = processor.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    logger.info(f"Loading PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()

    ## --- KEY CHANGE: Load TRAINING data instead of test data ---
    try:
        logger.info(f"Loading TRAINING data from {args.data_dir} to find hard samples.")
        train_df, _, _ = load_datasets_from_directory(args.data_dir) # We only need the training set
        if train_df.empty:
            logger.error("No training data found. Exiting."); return
        
        train_df['label'] = train_df['label'].astype(str).str.strip()
        train_df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
        all_labels = sorted(list(set(train_df['label'].tolist())))
        all_labels_str = ", ".join([f"'{label}'" for label in all_labels])
    except Exception as e:
        logger.error(f"Error loading data: {e}"); return

    # 3. Create Prompts for the Training Set
    prompts = []
    for _, row in train_df.iterrows():
        # This prompt structure is identical to your original script
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
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    # 4. Generate Predictions for the entire training set
    predictions = generate_predictions(model, processor, prompts, args.batch_size, all_labels=all_labels)
    true_labels = train_df['label'].tolist()

    ## --- KEY CHANGE: Identify and save the misclassified samples ---
    train_df['predicted_label'] = predictions
    
    # Filter the DataFrame to keep only rows where the prediction was incorrect
    hard_samples_df = train_df[train_df['label'] != train_df['predicted_label']].copy()
    
    # Optional: also include samples where the model failed to generate a valid label
    failed_parsing_df = train_df[train_df['predicted_label'] == 'n/a']
    hard_samples_df = pd.concat([hard_samples_df, failed_parsing_df]).drop_duplicates().reset_index(drop=True)

    logger.info(f"\n--- Analysis Complete ---")
    logger.info(f"Total training samples analyzed: {len(train_df)}")
    logger.info(f"Found {len(hard_samples_df)} misclassified or failed samples.")
    
    logger.info(f"Saving these 'hard samples' to: {args.output_file}")
    hard_samples_df.to_json(args.output_file, orient='records', lines=True, force_ascii=False)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify misclassified samples from a training set.")
    parser.add_argument("--base_model_name", type=str, default="google/gemma-2-2b-it", help="Base model name.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained PEFT adapter.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--output_file", type=str, default="hard_samples_for_cot.jsonl", help="File to save the misclassified samples.")
    
    cli_args = parser.parse_args()
    main(cli_args)