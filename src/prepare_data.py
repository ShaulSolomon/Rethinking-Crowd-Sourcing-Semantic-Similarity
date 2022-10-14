"""
This file should be run from SOTA project root folder

do 
> git clone https://github.com/brmson/dataset-sts
inside data folder


"""

import sys
import os
import glob
import pandas as pd
import pickle

sys.path.insert(0, os.path.join(os.getcwd(), "data\\dataset-sts"))


import pysts
from pysts.loader import load_sts


#s0, s1, labels = load_sts("data/dataset-sts/data/sts/semeval-sts/2015/headlines.test.tsv")
s0, s1, labels = load_sts("data/dataset-sts/data/sts/semeval-sts/all\../2012/OnWN.test.tsv")

print(f"Sentence A: {s0[0]}")
print(f"Sentence B: {s1[0]}")
print(f"Label: {labels[0]}")

#################
#   LOAD STS    #
#################


files = glob.glob("data/dataset-sts/data/sts/semeval-sts/all/*.test.tsv")
#files = glob.glob("dataset-sts/data/sts/semeval-sts/2015/head*.test.tsv")

data_df = pd.DataFrame()


for f in files:

    dataset_name = os.path.basename(f)
    print(dataset_name)
    
    fsize = os.stat(f).st_size
    if fsize < 40:
        a = open(f, "r").read()
        f = os.path.join(os.path.dirname(f), a)
   
    s0, s1, labels = load_sts(f)

    for i in range(len(s0)):
        st0 = [" ".join(words) for words in [s0[i]]][0]
        st1 = [" ".join(words) for words in [s1[i]]][0]
      
      #st0 = s0[i]
      #st1 = s1[i]

        lbl = labels[i]
        data_df = data_df.append({'text_1': st0, 'text_2': st1, 'label': lbl, 'dataset': dataset_name},
                                 ignore_index=True)

#######################
# ADDITIONAL DATASETS #
#######################
    
ds_files = ['Bible.csv',
            'Bible_Random.csv',
            'Paralex.csv',
            'Paralex_Random.csv',
            'Paraphrase.csv',
            'Paraphrase_Random.csv',
            'Yelp.csv',
            'Yelp_Random.csv']

for f in ds_files:
    tmp = pd.read_csv("data/"+f)
    f = f.replace('.csv', '')
    tmp['dataset'] = f
    data_df = pd.concat([data_df, tmp])

with open('data\combined_data.pickle', 'wb') as handle:
    pickle.dump(data_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#check
with open('data\combined_data.pickle', 'rb') as handle:
    b = pickle.load(handle)
