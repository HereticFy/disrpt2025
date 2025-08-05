# Environment Setup
1: Make sure you have Conda on your computer.

2: Use the command combined with the environment.yml file in this folder: **conda env create -f environment.yml**


# Train model For Stage 1
python task3_sft1.py --data_dir /path/to/sharedtask/data/ --output_dir /path/to/store/model --batch_size 8 --grad_accum 2 --epochs 3

Increase or decrease batch size and gradient accumulation in case you get OOM errors. The code is opitmized with flash attention for a newer GPU like H200.

# Train model For Stage 2
python task3_sft2.py --stage1_model_path path/to/stage1/final_model/ --rationale_path cots_train_data_qwen2.5_72b_hard_samples_COMPLETE.jsonl --epochs 1 --learning_rate 3e-5 --data_dir /path/to/sharedtask/data/

Link to the rationale file - url will go here

# Get Predictions
python task3_pred.py --stage1_path path/to/stage1/final_model/ --stage2_path path/to/stage2/final_model/ --data_dir /path/to/sharedtask/data --merge_for_inference

The script also automatically calls disrpt_eval.py script in the utils folder. The file path is hard coded to ``"../sharedtask2025/utils/disrpt_eval_2024.py"``. If it needs to be changed, change **line 116** and **line 236** in the file. In case you want the predictions, it will be stored in the ``output_dir`` of stage 2 model under the folder name ``predictions``.


# Additonal Files
``preprocess.py`` - This file has necessary functions that is used extensively by the training files like loading and parsing `.rels` files.

``generate_cot_vllm.py`` - This file is used to generate the rationales from the teacher model **Qwen/Qwen2.5-72B-Instruct**. It uses the library ``vllm`` which would have been installed with the environment.

``load_hard.py`` - This file loads the fine tuned model from Stage 1 and makes it infer on the training set to see what are the misclassified or hard samples. These samples are the ones passed to the teacher model in ``generate_cot_vllm.py`` for CoT generation. 

``test_stage1_model.py`` - Just to get a sanity check on the finetuned model from Stage 1, run this to check the micro-avg accuracy and F1-score on the test set.