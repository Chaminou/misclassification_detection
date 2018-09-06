Data error detection
=
Setup
-

```BASH
mkdir raw_data clear_data selected_data log model
pip3 install -r dependances.txt
```
Processing data
-

Place your training and testing xlsx in raw_data/ <br >
Let supposes my files are training_input.xlsx and testing_input.xlsx <br >
```BASH
python3 process_data.py -f training_input.xlsx -n clear_training_data
python3 process_data.py -f testing_input.xlsx -n clear_testing_data
```

Selecting data
-
Set the numbers of categories you want to use in line 109 of `data_selector.py` <br />
`data_selector.py` requires to run `process_data.py` first <br />
```BASH
python3 data_selector.py -f clear_training_data -t train -n selected_training_data
python3 data_selector.py -f clear_testing_data -t test -n selected_testing_data
```
NOTE : you must run first the training version before running the testing version (for dictionary reason) <br >

Training on dataset
-
`trainer.py` requires to run `data_selector.py` first <br >
```BASH
python3 trainer.py -train selected_training_data -test selected_testing_data
```
In `training.py`, choice return the index of the classification. You can find the `index to ID-Famille` dictionary at `selected_data/reverse_dictionary`. You can load it with load_df_pickle (pickle file) <br >
