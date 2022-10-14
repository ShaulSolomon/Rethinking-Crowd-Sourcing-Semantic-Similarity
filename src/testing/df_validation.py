import pandas as pd
import numpy as np

metrics = ['bleu', 
           'bleu1',
           'glove_cosine',
           'fasttext_cosine',
           'BertScore',
           'chrfScore',
           'POS Dist score',
           '1-gram_overlap',
           'ROUGE-1',
           'ROUGE-2',
           'ROUGE-l',
           'L2_score',
           'WMD']

def no_null(df):
    try:
        assert(df.isnull().any().any() == False)
    except:
        print("Has Null Values")
        find_null(df)
        return False
    return True

def no_inf(df):
    try:
        assert((np.inf not in df.values) == True)
    except:
        print("Has Infinite Values")
        return False
    return True

def find_null(df):
    null_columns = df.columns[df.isnull().any()]
    sum_null = df.isnull().sum().sum()
    print(f"Has {sum_null} null columns in: {null_columns}")
    display(df[df.isnull().any(axis=1)])

def all_metrics(df):
    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        print("Missing the following metrics: ", missing_metrics)
        return False
    return True

def all_tests(df):
    return no_null(df) & no_inf(df) & all_metrics(df)