import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
import logging
import json
import os
import re
import glob
from tqdm import tqdm
import time
from datetime import datetime
import gc
from vllm import LLM, SamplingParams  # <-- Import vLLM

# --- Configuration ---
HARD_SAMPLES_FILE = "./cots_train_data_qwen2.5_72b_hard_samples_vllm_full_failed.jsonl"
MODEL_NAME = "google/gemma-3-4b-it"  # For fine-tuning later
OUTPUT_DIR = "./discourse_model_gemma3_cot_h200"

# Teacher model config for generating rationales
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
SYNTHETIC_DATA_CACHE = "./cots_train_data_qwen2.5_72b_hard_samples_vllm_full1.jsonl"

# H200 Optimizations (still good practice)
torch.set_float32_matmul_precision('high')

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings and set TOKENIZERS_PARALLELISM
import warnings
warnings.filterwarnings("ignore", message="The 'use_cache' argument is deprecated")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- vLLM-Optimized Generation Function ---
def generate_cot_data_vllm(
    df: pd.DataFrame, 
    labels_list: list, 
    checkpoint_interval: int = 1000  # Process and save in chunks of this size
) -> pd.DataFrame:
    """
    Chain-of-Thought generation using vLLM for maximum throughput on a 4xH200 cluster.
    """
    logger.info(f"Initializing vLLM with {TEACHER_MODEL_NAME} on 4xH200...")

    # vLLM handles model distribution automatically across GPUs.
    # The `device_map` is replaced by `tensor_parallel_size`.
    llm = LLM(
        model=TEACHER_MODEL_NAME,
        tensor_parallel_size=4,  # <-- Key parameter for 4 GPUs
        dtype="bfloat16",
        gpu_memory_utilization=0.95, # Use 95% of GPU memory
        max_model_len=8192, # Max prompt length + max new tokens
        enforce_eager=True # Can improve stability for large models
    )
    logger.info("vLLM engine loaded successfully.")

    # Load the tokenizer just for creating the prompts
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    
    # Configure generation parameters using vLLM's SamplingParams
    # `do_sample=False` in HF is equivalent to `temperature=0` in vLLM.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=350,  # Corresponds to `max_new_tokens`
        repetition_penalty=1.02,
    )
    
    # Initialize the 'rationale' column
    df['rationale'] = pd.NA
    
    # Your detailed system prompt (no changes needed)
    system_prompt = """You are a distinguished computational linguist specializing in cross-linguistic discourse analysis. Your expertise spans discourse coherence theory, rhetorical structure theory (RST), and Penn Discourse TreeBank (PDTB) frameworks.

**ANALYTICAL TASK**: Generate precise, evidence-based Chain-of-Thought reasoning to explain discourse relations between text segments across multiple languages.

**DISCOURSE RELATION LABELS**: [{all_labels_str}]

**CRITICAL CONTEXT**:
• Unit 1 & Unit 2: Discourse segments extracted from larger sentences (may be clauses, phrases, or full sentences)
• Unit Snt: Complete sentences containing each unit (essential for understanding broader context)
• Direction: 
  - '1>2': Unit 1 initiates/sets up → Unit 2 responds/elaborates/resolves
  - '1<2': Unit 2 refers back to/builds upon → Unit 1

**ANALYTICAL FRAMEWORK FOR HIGH-QUALITY DISCOURSE ANALYSIS**:

1. **Linguistic Evidence Identification**:
   - Explicit discourse markers (connectives, adverbials, conjunctions)
   - Implicit semantic/pragmatic relationships
   - Syntactic structures indicating relations
   - Information structure (topic-comment, given-new)

2. **Cross-linguistic Awareness**:
   - Consider language-specific discourse patterns
   - Account for cultural pragmatic conventions
   - Note translation-invariant vs. language-specific features

3. **Comparative Label Analysis**:
   - Systematically compare the correct label with similar alternatives
   - Explain why competing labels are less appropriate
   - Use linguistic terminology precisely

4. **Reasoning Structure**:
   - Lead with the core relationship identification
   - Support with specific textual evidence
   - Conclude with comparative justification
   - Keep analysis concise (2-4 sentences) but comprehensive

**STYLE**: Write with the precision of academic discourse analysis while maintaining clarity. Use technical terms appropriately (e.g., "discourse connective", "anaphoric reference", "pragmatic implicature", "information structure")."""
    
    all_labels_str = ", ".join([f"'{label}'" for label in labels_list])
    system_prompt = system_prompt.format(all_labels_str=all_labels_str)
    
    # All your detailed few-shot examples (no changes needed)
    few_shot_examples = [
        # ... (your few-shot examples remain exactly the same) ...
        {
            "user": """Unit 1: "Longer maturities are thought to indicate declining interest rates"
Unit 2: "Shorter maturities are considered a sign of rising rates"
Unit 1 Snt: "Longer maturities are thought to indicate declining interest rates because they permit portfolio managers to retain relatively higher rates for a longer period."
Unit 2 Snt: "Shorter maturities are considered a sign of rising rates because portfolio managers can capture higher rates sooner."
Direction: 1<2
Label: contrast""",
            "assistant": "Both units discuss the same topic: the relationship between the length of a bond's maturity and the expected future direction of interest rates. The two units take this shared topic and present two opposing scenarios by changing the values of key variables which is further validated by the direction. There is no explicit discourse marker connecting the two units. The contrast is implicit and must be inferred by the reader. Hence, the correct label is `contrast`."
        },
        {
            "user": """Unit 1: "The asbestos fiber, crocidolite, is unusually resilient"
Unit 2: "once it enters the lungs,"
Unit 1 Snt: "The asbestos fiber, crocidolite, is unusually resilient once it enters the lungs, with even brief exposures to it causing symptoms that show up decades later, researchers said."
Unit 2 Snt: "The asbestos fiber, crocidolite, is unusually resilient once it enters the lungs, with even brief exposures to it causing symptoms that show up decades later, researchers said."
Direction: 1<2
Label: temporal""",
            "assistant": "The discourse connective `once` in Unit 2 indicates specific moment in time that serves as a starting point for another event or state. Unit 1 describes a state and Unit 2 describes a trigger event which is validated by the direction. The word `once` also strongly implies `condition` label. However, since it explicitly refers to a point in time, the `temporal` label is the best."
        },
        {
            "user": """Unit 1: "Whatever one thinks of the idea,"
Unit 2: "it's far more than the budgetary gimmick"
Unit 1 Snt: "Whatever one thinks of the idea, it's far more than the budgetary gimmick it may seem at first glance."
Unit 2 Snt: "Whatever one thinks of the idea, it's far more than the budgetary gimmick it may seem at first glance."
Direction: 1>2
Label: comment""",
            "assistant": "Unit 2 is an independent clause that makes a central claim while Unit 1 is an introductory clause to frame the central claim as a subjective take. This is validated by the direction of the relation. One might argue the relation is `concession`, as the author is conceding that others may think differently. While the structure is concessive, the overall pragmatic function of the entire utterance is to deliver the author's personal judgment and thus `comment` is the correct label."
        },
        {
            "user": """Unit 1: "do we start"
Unit 2: "Not yet"
Unit 1 Snt: "do we start"
Unit 2 Snt: "Not yet Wait for the 4th person"
Direction: 1>2
Label: query""",
            "assistant": "This interaction is classified as a query because Unit 1 is asking a direct question (`do we start`) to resolve an immediate uncertainty about a plan of action. This is also indicated by the direction. Unit 2 provides a direct answer (`Not yet`) that resolves this specific information gap. This label is more precise because it captures the specific purpose of checking a status before proceeding."
        },
        {
            "user": """Unit 1: "Elle a acheté des fleurs"
Unit 2: "pour décorer la maison"
Unit 1 Snt: "Elle a acheté des fleurs pour décorer la maison pour la fête."
Unit 2 Snt: "Elle a acheté des fleurs pour décorer la maison pour la fête."
Direction: 1>2
Label: purpose""",
            "assistant": "The preposition 'pour' in Unit 2 explicitly marks purpose in French, introducing the goal of the action in Unit 1. Direction 1>2 confirms Unit 1 states the action and Unit 2 provides its purpose. While 'pour' can indicate beneficiary ('for someone'), the infinitive 'décorer' clearly establishes purposive rather than benefactive reading. The `purpose` label precisely captures this goal-oriented relationship."
        }
    ]
    
    # --- Checkpoint and Prompt Preparation ---
    checkpoint_file = SYNTHETIC_DATA_CACHE.replace('.jsonl', '_checkpoint.jsonl')
    start_idx = 0
    all_rationales = [""] * len(df)  # Pre-allocate list
    
    if os.path.exists(checkpoint_file):
        logger.info(f"Found checkpoint file: {checkpoint_file}")
        checkpoint_df = pd.read_json(checkpoint_file, lines=True, orient='records')
        start_idx = len(checkpoint_df)
        logger.info(f"Resuming from index {start_idx}")
        all_rationales[:start_idx] = checkpoint_df['rationale'].tolist()

    logger.info("Pre-generating all prompts...")
    all_prompts = []
    for idx in tqdm(range(len(df)), desc="Preparing prompts"):
        row = df.iloc[idx]
        user_prompt = f"""Unit 1: "{row['unit1_txt']}"
Unit 2: "{row['unit2_txt']}"
Unit 1 Snt: "{row["unit1_snt"]}"
Unit 2 Snt: "{row["unit2_snt"]}"
Direction: {row["dir"]}
Label: {row['label']}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *sum([[{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["assistant"]}] for ex in few_shot_examples], []),
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)
    
    max_len = llm.llm_engine.model_config.max_model_len
    logger.info(f"Validating prompts against max model length of {max_len} tokens.")
    
    prompts_to_process = []
    # This list will map the index in `prompts_to_process` back to the original df index
    original_indices_map = [] 
    
    for i, prompt in enumerate(tqdm(all_prompts, desc="Validating prompt lengths")):
        # Only validate prompts that haven't been processed from a checkpoint
        if all_rationales[i]:
             continue

        # Check token length using the tokenizer
        # This is the proactive check to prevent the error
        if len(tokenizer.encode(prompt)) >= max_len:
            logger.warning(f"Skipping prompt at index {i}: length exceeds model maximum ({max_len}).")
            all_rationales[i] = "SKIPPED_PROMPT_TOO_LONG"
        else:
            prompts_to_process.append(prompt)
            original_indices_map.append(i)

    logger.info(f"Found {len(all_rationales) - sum(1 for r in all_rationales if r)} valid prompts to process.")
    
    # --- vLLM Generation with Checkpointing ---
    logger.info(f"Generating rationales for remaining samples...")
    start_time = time.time()
    
    # vLLM handles batching internally. We process in large chunks for checkpointing.
    num_remaining = len(all_prompts) - start_idx
    with tqdm(total=len(prompts_to_process), desc="Generating Rationales") as pbar:
        for i in range(0, len(prompts_to_process), checkpoint_interval):
            chunk_end = min(i + checkpoint_interval, len(prompts_to_process))
            prompt_chunk = prompts_to_process[i:chunk_end]
            
            if not prompt_chunk:
                break
                
            try:
                outputs = llm.generate(prompt_chunk, sampling_params)
                
                # Process and store results using the index map
                for j, output in enumerate(outputs):
                    # Get the original index from our map
                    original_index = original_indices_map[i + j]
                    rationale = output.outputs[0].text.strip()
                    
                    # Clean the output as before
                    if "Unit 1:" in rationale:
                        rationale = rationale.split("Unit 1:")[0].strip()
                    if "User:" in rationale:
                        rationale = rationale.split("User:")[0].strip()
                        
                    all_rationales[original_index] = rationale
                
                # Save checkpoint after processing the chunk
                # current_checkpoint_idx = chunk_end
                df['rationale'] = all_rationales 
                df.to_json(
                    checkpoint_file, 
                    orient='records', 
                    lines=True, 
                    force_ascii=False
                )
                # logger.info(f"Checkpoint saved at {current_checkpoint_idx} samples.")
                
            except Exception as e:
                # This will now only catch other errors (like OOM), not length errors.
                logger.error(f"A non-length-related error occurred in chunk starting at index {i}. Error: {e}")
                # Mark the failed items from this chunk
                for j in range(len(prompt_chunk)):
                    original_index = original_indices_map[i + j]
                    all_rationales[original_index] = "GENERATION_FAILED_RUNTIME_ERROR"
            
            pbar.update(len(prompt_chunk))
            # Show speed statistics
            elapsed = time.time() - start_time
            samples_processed = pbar.n
            speed = samples_processed / elapsed if elapsed > 0 else 0
            eta = (num_remaining - samples_processed) / speed if speed > 0 else 0
            pbar.set_postfix({
                'speed': f'{speed:.2f} samples/s',
                'eta': f'{eta/60:.1f} min'
            })

    # Add all generated rationales to the DataFrame
    df['rationale'] = all_rationales
    
    # Final statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in all_rationales if r and r != "GENERATION_FAILED")
    logger.info(f"Generation complete!")
    logger.info(f"Generated {successful}/{len(df)} rationales successfully")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    if total_time > 0:
      logger.info(f"Average speed: {num_remaining/total_time:.2f} samples/second")
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file removed after successful completion")
    
    # Cleanup
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return df

def main():
    try:
        logger.info(f"Loading hard samples from: {HARD_SAMPLES_FILE}")
        train_df = pd.read_json(HARD_SAMPLES_FILE, lines=True, orient='records')
        if train_df.empty:
            logger.error(f"No data found in {HARD_SAMPLES_FILE}. Exiting.")
            return
    except FileNotFoundError:
        logger.error(f"Could not find the hard samples file at: {HARD_SAMPLES_FILE}")
        logger.error("Please run the `find_hard_samples.py` script first to generate it.")
        return
    except Exception as e:
        logger.error(f"Error loading hard samples file: {e}")
        return

    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Get unique labels
    labels = train_df['label'].fillna('').astype(str).unique().tolist()
    labels = [l for l in labels if l]  # Remove empty strings
    logger.info(f"Found {len(labels)} unique discourse relation labels")
    
    # Use the vLLM-optimized generation function
    completed_df = generate_cot_data_vllm(
        train_df, 
        labels,
        checkpoint_interval=2048  # Process and save in larger chunks for efficiency
    )
    
    # Final validation
    successful_rationales = completed_df['rationale'].notna() & (completed_df['rationale'] != "GENERATION_FAILED")
    logger.info(f"Final validation: {successful_rationales.sum()}/{len(completed_df)} successful rationales")
    
    # Save the final results
    logger.info(f"Saving completed data to {SYNTHETIC_DATA_CACHE}")
    completed_df.to_json(SYNTHETIC_DATA_CACHE, orient='records', lines=True, force_ascii=False)
    logger.info("Script finished successfully!")
    
    # Save failed samples separately if any
    failed_df = completed_df[~successful_rationales]
    if len(failed_df) > 0:
        failed_file = SYNTHETIC_DATA_CACHE.replace('.jsonl', '_failed.jsonl')
        failed_df.to_json(failed_file, orient='records', lines=True, force_ascii=False)
        logger.warning(f"Saved {len(failed_df)} failed samples to {failed_file}")

if __name__ == "__main__":
    main()