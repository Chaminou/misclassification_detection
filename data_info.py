
import pickle
import pandas as pd
import tqdm

def load_df_pickle(name) :
    with open(name + '.pkl', 'rb') as f :
        df = pickle.load(f)
    return df

def count_categories(data) :
    df = data
    categories = []
    for i in range(len(df)) :
        id_famille = str(df['ID-Famile'][i])
        if id_famille not in categories :
            categories.append(id_famille)
    return len(categories)

df = load_df_pickle('df')
dfr = load_df_pickle('df-ready')

corpusdf = []
corpusdfr = []

for i in tqdm.tqdm(range(len(df))) :
    descriptiondf = str(df['LongDescription'][i])
    descriptiondfr = str(dfr['LongDescription'][i])
    descriptiondf_split = descriptiondf.lower().split(' ')
    descriptiondfr_split = descriptiondfr.lower().split(' ')
    for word_index in range(len(descriptiondf_split)) :
        if descriptiondf_split[word_index] not in corpusdf :
            corpusdf.append(descriptiondf_split[word_index])
    for word_index in range(len(descriptiondfr_split)) :
        if descriptiondfr_split[word_index] not in corpusdfr :
            corpusdfr.append(descriptiondfr_split[word_index])

print('vocab size unprocessed data', len(corpusdf))
print('vocab size processed data', len(corpusdfr))

print('nb of categories', count_categories(df))
