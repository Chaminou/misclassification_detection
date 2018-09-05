first, create 5 folders : raw_data/ clear_data/ selected_data/ log/ model/ <br >
place your training and testing xlsx in raw_data/ <br >
let supposes my files are training_input.xlsx and testing_input.xlsx in raw_data/
to run process_data.py, use the following pattern : "python3 process_data.py -f training_input.xlsx -n clear_training_data" <br >
do that for both files <br >
then select the numbers of categories you want to use in line 109 ofdata_selector.py <br />
data_selector requires to run process_data first <br />
to run data_selector.py, use the following pattern : "python3 data_selector.py -f clear_training_data -t train -n selected_training_data" <br >
you must run first the training version before running the testing version (for dictionary reason) <br >
do that for both files <br >
trainer requires to run data_selector first <br >
to run trainer.py use the following pattern : "python3 trainer.py -train selected_training_data -test selected_testing_data" <br >
in training, choice return the index of the classification. You can find the index to ID-Famille dictionary here : selected_data/reverse_dictionary. You can load it with load_df_pickle (pickle file) <br >
