import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import your data loading functions ---
# Make sure preprocess.py is accessible
from preprocess import rel_reader1
from task3_sft_stage1 import load_data_from_directory

SYSTEM_PROMPT = """You are a discourse relation classifier. Your task is to analyze text pairs and classify their discourse relationship and label them from the given labels.

IMPORTANT: Your response must be ONLY a JSON object with the format {"label": "your_classification"}
Do not include any other text or explanations outside of the JSON."""

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

def main(args):
    # 1. Load Model and Processor
    logger.info(f"Loading base model: {args.base_model_name}")
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation= "flash_attention_2" if torch.cuda.is_available() else "eager",
        token=True, # Or your Hugging Face token
    )

    logger.info(f"Loading PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload() # Merge adapter for faster inference

    # 2. Load and Prepare Test Data
    try:
        _, _, test_df = load_data_from_directory(args.data_dir)
        if test_df.empty:
            logger.error("No test data found. Exiting.")
            return
        
        test_df['label'] = test_df['label'].astype(str).str.strip()
        test_df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
        all_labels = sorted(list(set(test_df['label'].tolist())))
        all_labels_str = ", ".join([f"'{label}'" for label in all_labels])
    except Exception as e:
        logger.error(f"Error loading data: {e}"); return

    # 3. Create Prompts for Test Set
    prompts = []
    for _, row in test_df.iterrows():
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

        messages = [
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    # 4. Generate Predictions
    predictions = generate_predictions(model, processor, prompts, args.batch_size, all_labels=all_labels)
    true_labels = test_df['label'].tolist()

    # 5. Calculate and Report Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted', labels=all_labels, zero_division=0)

    logger.info(f"\n--- Intermediate Test Set Results (Stage 1 Model) ---")
    logger.info(f"  Overall Accuracy: {accuracy:.4f}")
    logger.info(f"  Overall Weighted F1-Score: {f1:.4f}")
    logger.info(f"Failed predictions (n/a): {predictions.count('n/a')}/{len(predictions)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 fine-tuned model.")
    parser.add_argument("--base_model_name", type=str, default="google/gemma-2-2b-it", help="Base model name from Hugging Face.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained PEFT adapter checkpoint (e.g., './discourse_gemma_2_stage1/checkpoint-1234').")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    
    cli_args = parser.parse_args()
    main(cli_args)