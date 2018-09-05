
import pickle
import pandas as pd
import tqdm
import argparse
import os

def save_df_pickle(df, name) :
    with open(name, 'wb') as f :
        pickle.dump(df, f)

#load dataframe with pickle
def load_df_pickle(name) :
    with open(name, 'rb') as f :
        df = pickle.load(f)
    return df

def count_categories(data) :
    df = data
    categories = []
    for i in range(len(df)) :
        id_famille = str(df['ID-Famile'][i])
        if id_famille not in categories :
            categories.append(id_famille)
    print(len(categories))

def dico_categories(data) :
    print('##### CREATE CATEGORIES DICT #####')
    df = data
    categories = {}
    for i in tqdm.tqdm((range(len(df)))) :
        id_famille = str(df['ID-Famile'][i])
        if id_famille not in categories :
            categories[id_famille] = 0
        categories[id_famille] += 1
    return categories

def get_n_biggest_categories(dico, n) :
    print('##### GET N BIGGEST CATEGORIES #####')
    list_of_categories = []
    for (k, v) in sorted(dico.items(), key=lambda x:x[1])[-n:] :
        list_of_categories.append(k)
    return list_of_categories

def create_selected_categories_df(data, list_of_categories) :
    print('##### CREATE THE NEW DATAFRAME #####')
    df = data
    selected_df = pd.DataFrame(columns=('LongDescription', 'ID-Famile'))

    dico_categories = {}
    for index in range(len(list_of_categories)) :
        dico_categories[list_of_categories[index]] = index
    reverse_dico = {v: k for k, v in dico_categories.items()}

    counter_of_selected_item = 0
    for i in tqdm.tqdm(range(len(df))) :
        if str(df['ID-Famile'][i]) in list_of_categories :
            selected_df.loc[counter_of_selected_item] = [df['LongDescription'][i], dico_categories[str(df['ID-Famile'][i])]]
            counter_of_selected_item += 1

    return selected_df, reverse_dico


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Select data')

    parser.add_argument('-f', dest='file', default=None,
                       help='pickled dataframe')
    parser.add_argument('-t', dest='type', default=None,
                       help='type of data : training or testing')
    parser.add_argument('-n', dest='name', default=None,
                       help='name of the pickled output dataframe with selected categories')

    args = parser.parse_args()

    is_file_valid = False
    is_type_valid = False
    is_name_valid = False

    if args.file != None :
        if os.path.isfile("clear_data/" + args.file) :
            print("valid dataframe")
            is_file_valid = True
        else :
            print('no file detected')
    else :
        print('please set an input file with -f')

    if args.type != None :
        if args.type in ['test', 'train'] :
            print("valid type")
            is_type_valid = True
        else :
            print('unknown type')
    else :
        print('please set a type with -t')

    if args.name != None :
        print('pickled dataframe will be saved as ' + args.name)
        is_name_valid = True
    else :
        print('please set a name with -n')

    if is_file_valid and is_type_valid and is_name_valid:
        df = load_df_pickle('clear_data/' + args.file)
        if args.type == 'train' :
            dico = dico_categories(df)
            list_of_categories = get_n_biggest_categories(dico, 20)
            save_df_pickle(list_of_categories, 'selected_data/list_of_categories')
        if args.type == 'test' :
            list_of_categories = load_df_pickle('selected_data/list_of_categories')

        working_df, reverse_dico = create_selected_categories_df(df, list_of_categories)
        save_df_pickle(working_df, 'selected_data/' + args.name)
        save_df_pickle(reverse_dico, 'selected_data/reverse_dictionary')
