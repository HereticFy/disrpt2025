import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import subprocess
import glob

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import your data loading functions ---
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
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                eos_token_id=processor.eos_token_id,  
                pad_token_id=processor.pad_token_id,
                # temperature=0.1,
                do_sample=False
            )
        
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_outputs = processor.batch_decode(generated_tokens, skip_special_tokens=True)

        for output in decoded_outputs:
            pred = "n/a"
            output = output.strip()
            
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
                
                if pred == "n/a":
                    logger.warning(f"Failed to extract label from output: {output[:100]}...")
            
            predictions.append(pred)
    return predictions

def save_predictions_as_rels(test_df, predictions, output_path):
    """Save predictions in DISRPT .rels format for evaluation"""
    # DISRPT .rels header
    header = "doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\tu1_raw\tu2_raw\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\trel_type\torig_label\tlabel"
    
    lines = [header]
    
    for idx, (_, row) in enumerate(test_df.iterrows()):
        # Extract needed fields, handling missing values
        doc = row.get('doc', 'unknown_doc')
        unit1_toks = row.get('unit1_toks', '')
        unit2_toks = row.get('unit2_toks', '')
        unit1_txt = row.get('unit1_txt', '')
        unit2_txt = row.get('unit2_txt', '')
        u1_raw = row.get('u1_raw', row.get('unit1_txt', ''))
        u2_raw = row.get('u2_raw', row.get('unit2_txt', ''))
        s1_toks = row.get('s1_toks', '')
        s2_toks = row.get('s2_toks', '')
        unit1_sent = row.get('unit1_snt', row.get('unit1_sent', ''))
        unit2_sent = row.get('unit2_snt', row.get('unit2_sent', ''))
        dir_val = row.get('dir', '1>2')
        rel_type = row.get('rel_type', 'implicit')
        orig_label = row.get('orig_label', row.get('label', ''))
        
        # Use prediction for label
        pred_label = predictions[idx]
        
        # Create tab-separated line
        line = f"{doc}\t{unit1_toks}\t{unit2_toks}\t{unit1_txt}\t{unit2_txt}\t{u1_raw}\t{u2_raw}\t{s1_toks}\t{s2_toks}\t{unit1_sent}\t{unit2_sent}\t{dir_val}\t{rel_type}\t{orig_label}\t{pred_label}"
        lines.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path

def call_disrpt_eval(gold_path, pred_path, dataset_name):
    """Call the DISRPT evaluation script and parse results"""
    try:
        # Run the evaluation script
        cmd = [
            'python', '../sharedtask2025/utils/disrpt_eval_2024.py',
            '-g', gold_path,
            '-p', pred_path,
            '-t', 'R'  # Relations task
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Evaluation script failed for {dataset_name}: {result.stderr}")
            return None
        
        # Parse JSON output
        eval_results = json.loads(result.stdout)
        print(eval_results)
        logger.info(f"\n{'='*60}")
        logger.info(f"DISRPT EVALUATION RESULTS - {dataset_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {eval_results['labels_accuracy']:.4f}")
        logger.info(f"Gold count: {eval_results['labels_gold_count']}")
        logger.info(f"Pred count: {eval_results['labels_pred_count']}")
        
        # if 'labels_classification_report' in eval_results:
        #     logger.info("\nPer-label Classification Report:")
        #     report = eval_results['labels_classification_report']
        #     for label, metrics in report.items():
        #         if label not in ['accuracy', 'macro avg', 'weighted avg']:
        #             logger.info(f"  {label}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        logger.info(f"{'='*60}\n")
        
        return eval_results
        
    except Exception as e:
        logger.error(f"Error running evaluation for {dataset_name}: {e}")
        return None

def process_dataset(model, processor, dataset_dir, predictions_dir, batch_size, all_labels_str):
    """Process a single dataset directory"""
    dataset_name = os.path.basename(dataset_dir)
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*70}")
    
    # Find test file
    test_files = glob.glob(os.path.join(dataset_dir, "*_test.rels"))
    if not test_files:
        logger.warning(f"No test file found in {dataset_dir}")
        return None
    
    test_file = test_files[0]
    logger.info(f"Found test file: {test_file}")
    
    # Load test data using rel_reader1
    # try:
    content = rel_reader1(test_file)
    test_df = pd.DataFrame(content, columns=['doc', 'unit1_toks', 'unit1_txt', 'unit2_toks', 'unit2_txt', 'unit1_snt', 'unit2_snt', 'dir', 'label'])
            
    if test_df.empty:
        logger.error(f"No data loaded from {test_file}")
        return None
    
    test_df['label'] = test_df['label'].astype(str).str.strip()
    test_df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
    test_df = test_df[test_df['label'] != '']
    
    all_labels = sorted(list(set(test_df['label'].tolist())))
    
    logger.info(f"Test set size: {len(test_df)}")
    # logger.info(f"Number of unique labels: {len(all_labels_str)}")
        
    # except Exception as e:
    #     logger.error(f"Error loading data from {dataset_dir}: {e}")
    #     return None
    
    # Create prompts
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

        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(model, processor, prompts, batch_size, all_labels=all_labels)
    true_labels = test_df['label'].tolist()
    
    # Calculate sklearn metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted', labels=all_labels, zero_division=0)
    
    # logger.info(f"\n{'='*60}")
    # logger.info(f"SKLEARN METRICS - {dataset_name}")
    # logger.info(f"{'='*60}")
    # logger.info(f"Accuracy: {accuracy:.4f}")
    # logger.info(f"Weighted F1-Score: {f1:.4f}")
    # logger.info(f"Failed predictions (n/a): {predictions.count('n/a')}/{len(predictions)}")
    
    # Save predictions
    pred_path = os.path.join(predictions_dir, f"{dataset_name}_pred.rels")
    save_predictions_as_rels(test_df, predictions, pred_path)
    logger.info(f"Predictions saved to: {pred_path}")
    
    # Run DISRPT evaluation
    disrpt_results = None
    if os.path.exists('../sharedtask2025/utils/disrpt_eval_2024.py'):
        disrpt_results = call_disrpt_eval(test_file, pred_path, dataset_name)
    else:
        logger.warning("disrpt_eval_2024.py not found. Skipping DISRPT evaluation.")
    
    # Return results
    results = {
        'dataset': dataset_name,
        'test_samples': len(test_df),
        'sklearn_metrics': {
            'accuracy': accuracy,
            'f1_score': f1,
            'failed_predictions': predictions.count('n/a'),
        }
    }
    
    if disrpt_results:
        results['disrpt_metrics'] = disrpt_results
    
    return results

def main(args):
    logger.info("="*70)
    logger.info("Stage 2 Model Evaluation - Multiple Datasets")
    logger.info("="*70)
    
    # 1. Load Model
    logger.info("Loading Stage 2 model components...")
    
    # Load processor from Stage 2
    stage2_model_path = os.path.join(args.stage2_path, "final_model") if "final_model" not in args.stage2_path else args.stage2_path
    processor = AutoProcessor.from_pretrained(stage2_model_path)
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token
    
    # Step 1: Load the base model
    logger.info(f"Step 1: Loading base model {args.base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Step 2: Load and merge Stage 1 LoRA
    logger.info(f"Step 2: Loading and merging Stage 1 from {args.stage1_path}...")
    stage1_model = PeftModel.from_pretrained(base_model, args.stage1_path)
    merged_base = stage1_model.merge_and_unload()
    
    # Step 3: Load Stage 2 LoRA
    logger.info(f"Step 3: Loading Stage 2 LoRA from {stage2_model_path}...")
    model = PeftModel.from_pretrained(merged_base, stage2_model_path)
    
    # Optional: merge for faster inference
    if args.merge_for_inference:
        logger.info("Merging Stage 2 adapter for faster inference...")
        model = model.merge_and_unload()
    
    logger.info("Model loading complete!")
    
    #get all labels
    _, _, test_df = load_data_from_directory(args.data_dir)
    test_df['label'] = test_df['label'].astype(str).str.strip()
    test_df.dropna(subset=['label', 'unit1_txt', 'unit2_txt'], inplace=True)
    test_df = test_df[test_df['label'] != '']
    
    all_labels = sorted(list(set(test_df['label'].tolist())))
    all_labels_str = ", ".join([f"'{label}'" for label in all_labels])

    # 2. Create predictions directory
    predictions_dir = os.path.join(os.path.dirname(args.stage2_path), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    logger.info(f"Predictions will be saved to: {predictions_dir}")
    
    # 3. Process each dataset
    all_results = []
    
    if args.dataset:
        # Process specific dataset
        dataset_dirs = [os.path.join(args.data_dir, args.dataset)]
    else:
        # Process all datasets
        dataset_dirs = sorted(glob.glob(os.path.join(args.data_dir, "*")))
        dataset_dirs = [d for d in dataset_dirs if os.path.isdir(d)]
    
    logger.info(f"Found {len(dataset_dirs)} dataset(s) to process")
    
    for dataset_dir in dataset_dirs:
        if not os.path.isdir(dataset_dir):
            continue
            
        results = process_dataset(model, processor, dataset_dir, predictions_dir, args.batch_size, all_labels_str)
        if results:
            all_results.append(results)
    
    # 4. Save combined results
    combined_results = {
        'model_info': {
            'stage1_path': args.stage1_path,
            'stage2_path': args.stage2_path,
            'base_model': args.base_model_name
        },
        'results_by_dataset': all_results,
        'summary': {}
    }
    
    # Calculate summary statistics
    if all_results:
        avg_accuracy = sum(r['sklearn_metrics']['accuracy'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['sklearn_metrics']['f1_score'] for r in all_results) / len(all_results)
        total_samples = sum(r['test_samples'] for r in all_results)
        
        combined_results['summary'] = {
            'total_datasets': len(all_results),
            'total_samples': total_samples,
            'average_accuracy': avg_accuracy,
            'average_f1_score': avg_f1
        }
        
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total datasets evaluated: {len(all_results)}")
        logger.info(f"Total test samples: {total_samples}")
        logger.info(f"Average accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average F1-score: {avg_f1:.4f}")
        logger.info(f"{'='*70}")
    
    # Save all results
    results_path = os.path.join(predictions_dir, "all_results.json")
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"\nAll results saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 fine-tuned model on test sets.")
    parser.add_argument("--base_model_name", type=str, default="google/gemma-2-2b-it", 
                       help="Base model name from Hugging Face")
    parser.add_argument("--stage1_path", type=str, required=True, 
                       help="Path to Stage 1 model (e.g., ./discourse_gemma_stage1/final_model)")
    parser.add_argument("--stage2_path", type=str, required=True, 
                       help="Path to Stage 2 model (e.g., ./discourse_gemma_stage2)")
    parser.add_argument("--data_dir", type=str, default="./data", 
                       help="Directory containing the dataset folders")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to evaluate (e.g., 'eng.pdtb.pdtb'). If not specified, evaluates all datasets.")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for evaluation")
    parser.add_argument("--merge_for_inference", action="store_true", 
                       help="Merge adapters for faster inference")
    
    args = parser.parse_args()
    main(args)