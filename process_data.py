
import pandas as pd
import pickle
import numpy as np
import string
import tqdm
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords


#save dataframe with pickle
def save_df_pickle(df, name) :
    with open(name + '.pkl', 'wb') as f :
        pickle.dump(df, f)

#load dataframe with pickle
def load_df_pickle(name) :
    with open(name + '.pkl', 'rb') as f :
        df = pickle.load(f)
    return df

#load df from xlsx
def load_df_excel(name) :
    df = pd.read_excel(name)
    return df


def clean_columns(data) :
    print('##### CLEAN COLUMNS #####')
    df = data
    #interation over the whole dataset to fill up NaN in some categories in 'ID-Famile' and 'Famille'
    for i in tqdm.tqdm(range(len(df))) :
        # df['Famille'][i] and df['ID-Famile'][i] are always the same if one is NaN, so just checking one
        if pd.isna(df['Famille'][i]) :
            df['ID-Famile'][i] = df['ID-sousCategorie'][i]
            df['Famille'][i] = df['sousCategorie'][i]
        if pd.isna(df['LongDescription'][i]) :
            df['LongDescription'][i] = df['ShortDescription'][i]
    #saveing dataframe
    return df

def replace_chars(data) :
    print('##### REPLACE CHARS #####')
    df = data
    replace_dico = {'&rsquo;':"'",
                    '&oelig;':'oe',
                    '&lsquo;':"",#les 4 suivants représente un '
                    '&ldquo':"",
                    '&rdquo;':"",
                    '%u2019':"",
                    'a§':'c',
                    'a«':'e',
                    'a¯':'i',
                    'a´':'o',
                    'à':'a',
                    'â':'a',
                    'ç':'c',
                    'é':'e',
                    'è':'e',
                    'ê':'e',
                    'ù':'u',
                    'û':'u',
                    '\xa0':'',
                    ',':'',
                    '.':'',
                    ')':'',
                    '(':'',
                    ':':'',
                    '!':'',
                    '©':'',
                    ';':'',
                    '?':'',
                    "'":'',
                    '"':'',}
    for i in tqdm.tqdm(range(len(df))) :
        description = str(df['LongDescription'][i])
        description_split = description.lower().split(' ')
        for word_index in range(len(description_split)) :
            for pattern in replace_dico :
                description_split[word_index] = description_split[word_index].replace(pattern, replace_dico[pattern])
        df['LongDescription'][i] = ' '.join(description_split)

    return df

def remove_words(data) :
    print('##### REMOVE WORDS #####')
    df = data
    list_of_removal_flags = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%', '°', '/', '-']
    for i in tqdm.tqdm(range(len(df))):
        description = str(df['LongDescription'][i])
        description_split = description.lower().split(' ')
        filtered_description = []
        for word_index in range(len(description_split)) :
            flaged = False
            for flag in list_of_removal_flags :
                if flag in description_split[word_index] :
                    flaged = True
            if not flaged :
                filtered_description.append(description_split[word_index])
        df['LongDescription'][i] = ' '.join(filtered_description)

    return df

def lemmatize_words(data) :
    print('##### LEMMATIZE WORDS #####')
    df = data
    stemmer = FrenchStemmer()
    for i in tqdm.tqdm(range(len(df))):
        description = str(df['LongDescription'][i])
        description_split = description.lower().split(' ')
        lemmatized_description = []
        for word_index in range(len(description_split)) :
            lemmatized_description.append(stemmer.stem(description_split[word_index]))
        df['LongDescription'][i] = ' '.join(lemmatized_description)

    return df

def remove_stop_words(data) :
    print('##### REMOVE STOP WORDS #####')
    df = data
    stop_words = ["l'", "d'", 'a', 'cm','m', 'est', 'tous', 'toutes', 'tout', 'dune', 'si', 'afin', 'cet', 'cette', 'ces', 'les'] + stopwords.words('french')
    for i in tqdm.tqdm(range(len(df))) :
        description = str(df['LongDescription'][i])
        description_split = description.lower().split(' ')
        stopword_filtered_description = []
        for word_index in range(len(description_split)) :
            is_stop_word = False
            for stop_word in stop_words :
                if description_split[word_index] == stop_word :
                    is_stop_word = True
            if not is_stop_word :
                stopword_filtered_description.append(description_split[word_index])
        df['LongDescription'][i] = ' '.join(stopword_filtered_description)

    return df

#Can spot undesired char in word and return them, super useful for data debugging
def print_error(data) :
    corpus = []
    counter = 0
    for i in range(len(df)) :
        description = str(df['LongDescription'][i])
        description_split = description.lower().split(' ')
        counter += len(description_split)
        for j in description_split :
            is_ok = True
            for k in j :
                if k not in string.ascii_lowercase :
                    is_ok = False
            if not is_ok :
                corpus.append(j)
    print(corpus)
    print(len(corpus))

def count_categories(data) :
    df = data
    categories = []
    for i in range(len(df)) :
        id_famille = str(df['ID-Famile'][i])
        if id_famille not in categories :
            categories.append(id_famille)
    print(len(categories))


if __name__ == '__main__' :
    #load excel as pandas dataframes
    df = load_df_excel('raw_data/training_input.xlsx')
    #copy information to make every columns ok to use
    df = clean_columns(df)
    #save that dataframe before processing
    save_df_pickle(df, 'clear_data/df')
    #load dataframe, can skip it here
    df = load_df_pickle('df')
    #removing word with some flags in them
    df = remove_words(df)
    #replacing some chars in words
    df = replace_chars(df)
    #removing stop words
    df = remove_stop_words(df)
    #lemmatize the remaining words
    df = lemmatize_words(df)
    #save the processed dataframe to an other location, ready to use in ML
    save_df_pickle(df, 'clear_data/df-processed')
