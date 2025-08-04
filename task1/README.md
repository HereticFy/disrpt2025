# Task 1

# 1: Install the enviroment
1: Make sure you have Conda on your computer

2: Use the command combined with the environment.yml file in this folder: **conda env create -f environment.yml**

# 2: Preprocessing datasets
1: For Task 1, you have to download the datasets and put them in the main repository. Note that you have to **make sure** all the corpora are complete! Because some of the corpora do not contain text!!!!!

2: Once you have all the datasets in the main repository, you will have two folders, task1 and data, in it.

3: Run the command

>python preprocessing.py

It will preprocess all the datasets you need for this task. After the whole process is done, you can see some new .json files generated in the datasets we need for Task 1. For example, in ./data/deu.rst.pcc, you can see three new .json files generated here.

If you are stuck in this step, please check lines 530-547 in the file preprocessing.py. You can modify the path according to where you put the data folder.

# 3: Training
Now, you need to run from run_script_1.sh to run_script_13.sh by using the following command:
>sh run_script_[numbers from 1 to 13].sh

Each script will train and test different datasets. You can refer to the following list:

1. run_script_1.sh -> deu.rst.pcc eng.sdrt.stac fas.rst.prstc nld.rst.nldt zho.dep.scidtb ces.rst.crdt eng.sdrt.msdc
2. run_script_2.sh -> eng.dep.scidtb rus.rst.rrt eng.etst.gentle
3. run_script_3.sh -> eng.erst.gum
4. run_script_4.sh -> eng.rst.rstdt
5. run_script_5.sh -> eus.rst.ert por.rst.cstn
6. run_script_6.sh -> fra.sdrt.annodis fra.sdrt.summre
7. run_script_7.sh -> spa.rst.rststb
8. run_script_8.sh -> spa.rst.sctb
9. run_script_9.sh -> zho.rst.gcdt
10. run_script_10.sh -> zho.rst.sctb
11. run_script_11.sh -> eng.dep.covdtb
12. run_script_12.sh -> eng.rst.umuc eng.rst.oll
13. run_script_13.sh -> eng.rst.sts

# 4: Evaluation
Once you have completed the training process, you will see a new folder called **training_results**. In this folder, you can find our training results stored in different folders, such as mt5-xl_script1_results.

Go into one of these folders, and you will see some .tok files generated for the evaluation. For example, you should see a deu.rst.pcc_test_pred.tok file in mt5-xl_script1_results.

Then, please use the code provided by the organisers to evaluate the predicted .tok file with the original .tok file.

In our setting, we put the utils folder provided by the organisers into the main repository. 

Then, we evaluated as follows:
>my_eval = SegmentationEvaluation(current_test_dataset, tok_file_name, pred_file, False, False)
> 
>my_eval.compute_scores()
> 
>my_eval.print_results()

The tok_file_name above is the path to the dataset's original test .tok file, and pred_file refers to the path of our generated .tok file. Thus, the key to success in this step is to compare the generated .tok file with the official test .tok file.

For more details, we would like to invite the reader to check the evaluation code file disrpt_eval_2024.py provided by the organisers.