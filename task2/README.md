# Environment Setup
1: Make sure you have Conda on your computer.

2: Use the command combined with the environment.yml file in this folder: **conda env create -f environment.yml**

# Running the code

## To replicate the results from the paper

python task2.py --use_focal_loss --ensemble_method attention --bf16 --tf32 --data_dir /path/to/sharedtask/data --output_dir output/directory/for/model 

Note that the flags ``--bf16`` and ``--tf32`` were specifically used in our case because we had access to newer GPU like H200. If BF16 mixed precision training is not supported on the GPU you are running on, please use ``--fp16`` instead of ``--bf16``. You can also remove or keep ``--tf32``. However, ``bf16`` and ``--fp16`` are mutually exclusive and cannot be used together. In general, the code is optimized for the newer GPUs.

The script also automatically calls disrpt_eval.py script in the utils folder. The file path is hard coded to ``"../sharedtask2025/utils/disrpt_eval_2024.py"``. You can change it in **line 1234**. In case you want the predictions, it will be stored in the ``output_dir`` under the folder name ``predictions``.



Please check out the other flags in the code for a more extensive experimentation. 