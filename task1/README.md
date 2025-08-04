# Task 1

# 1: Install the enviroment
1: Make sure you have Conda in your computer

2: Use the command combined with the enviroment.yml file in this folder: **conda env create -f environment.yml**

# 2: Preprocessing datasets
1: For Task 1, you have to download the datasets and put them in the main repository. Note that you have to **make sure** all the corpora is complete! Because some of the corpora do not contain text!!!!!

2: Once you put all the datasets in to the main repository, you now have two folders, task1 and data, in your main repository.

3: Run the command

>python preprocessing.py

It will preprocess all the datasets that you need for this task. After the whole process is done, you can see some new .json file generated in the datasets that we need for Task 1. For example, in ./data/deu.rst.pcc, you can see three new .json file generated here.

If you stuck in this step, please check the line 530-547 in the file preprocessing.py. You can modify the path according to where you put the data folder.

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
7. run_script_1.sh -> spa.rst.rststb
8. run_script_1.sh -> spa.rst.sctb
9. run_script_1.sh -> zho.rst.gcdt
10. run_script_1.sh -> zho.rst.sctb
11. run_script_1.sh -> eng.dep.covdtb
12. run_script_1.sh -> eng.rst.umuc eng.rst.oll
13. run_script_1.sh -> eng.rst.sts

# 4: Evaluation
Once you finished all the training process, you will see there's new folder created called **training_results**. In this folder, you can find our training results stored in different folders, for example: script1_results.

Go into one of these folders, you will see some .tok files generated for the evaluation. For example, you should see a deu.rst.pcc_test_pred.tok in script1_results.

Then, please use the code provided by the organizers to evaluate the predicted .tok file with the original .tok file.

In our setting, we put the utils folder provided by the organizers into the main repository 

Then, we evaluated as follows:
>my_eval = SegmentationEvaluation(current_test_dataset, tok_file_name, pred_file, False, False)
> 
>my_eval.compute_scores()
> 
>my_eval.print_results()

The tok_file_name above is the path to the original test .tok file of the dataset and pred_file refers to the path of our generated .tok file. Thus, the key to achieve success in this step is to compare the generated .tok file with the official test .tok file.

For more details, we would like to invite the reader to check the evaluation code file disrpt_eval_2024.py provided by the organizers.