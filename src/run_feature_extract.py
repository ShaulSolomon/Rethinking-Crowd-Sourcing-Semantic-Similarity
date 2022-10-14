"""
This file implements running the different semantic similiarity metrics on a dataset of paired sentences
"""

import sys
import nltk
from pathlib import Path
nltk.download('stopwords')
# adding cwd to path to avoid "No module named src.*" errors
MODULE_PATH = Path(__file__).resolve().parents[1].resolve()
sys.path.insert(0, str(MODULE_PATH))

import pickle
import argparse
import pandas as pd
from src.utils import get_environment_variables
from src.features.bleu import Bleu
from src.features.bertscore import BertScore
from src.features.chrFScore import chrFScore
from src.features.cosine_similarites import CosineSimilarity
from src.features.elmo_euclidean_distance import EuclideanElmoDistance
from src.features.ngram_overlap import NgramOverlap
from src.features.POS_distance import POSDistance
from src.features.ROUGE import ROUGE
from src.features.WMD import WMD
from src.preprocessing import text_preprocessing


def main(args):
    PATH_ROOT, PATH_DATA, GloVe_840B_300d_PATH, Glove_twitter_27B_PATH, ENV = get_environment_variables()
    picklefile = args.pickle
    stopwords = args.keep_stopwords == False

    if 'pickle' in picklefile:
        with open(picklefile, 'rb') as handle:
            df = pickle.load(handle)
    else:
        df = pd.read_csv(picklefile, index_col=0)

    txt_col_format = 'text_' if 'text_1' in df.columns else 'text'

    df.dropna(subset=[f'{txt_col_format}1', f'{txt_col_format}2'], inplace=True)

    df = df[(df[f'{txt_col_format}1'].str.strip().str.split().apply(len) >= 2) &
            (df[f'{txt_col_format}2'].str.strip().str.split().apply(len) >= 2)].copy()

    df[f'{txt_col_format}1'] = text_preprocessing(df[f'{txt_col_format}1'])
    df[f'{txt_col_format}2'] = text_preprocessing(df[f'{txt_col_format}2'])

    df.dropna(subset=[f'{txt_col_format}1', f'{txt_col_format}2'], inplace=True)

    extractors = dict(
        bleu=Bleu(txt_col_format, stopwords=stopwords),
        cosine_similarites=CosineSimilarity(val=txt_col_format, glove_path=Glove_twitter_27B_PATH),
        elmo=EuclideanElmoDistance(val=txt_col_format),
        bert=BertScore(val=txt_col_format),
        chrf_score=chrFScore(val=txt_col_format),
        pos_distance=POSDistance(val=txt_col_format, vector_path=Glove_twitter_27B_PATH),
        wmd=WMD(val=txt_col_format, vector_path=GloVe_840B_300d_PATH),
        ngram_overlap=NgramOverlap(args.max_n, val=txt_col_format),
        rouge=ROUGE(val=txt_col_format, stopwords=stopwords))

    features = args.features
    exclude = args.exclude
    if features == 'ALL':
        features = list(extractors.keys())
        if exclude:
            exclude = exclude.lower().split(',')
            for ex in exclude:
                features.remove(ex)
    else:
        features = features.lower().split(',')

    for feature_name, extractor in extractors.items():
        if feature_name in features:
            try:
                print(f'Running {feature_name} metric')
                df = extractor.run(df)
            except Exception as e:
                print(f'Threw error on ')
                print(f'Running {feature_name} generated {type(e)} with message "{e}"')

    if 'pickle' in picklefile:
        with open(picklefile, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df.to_csv(picklefile, index=True)


################################
# For Command-line running
################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, required=False,
                        default='data/datasets/Yelp_human.csv',
                        help='pickle path for combined dataset')
    parser.add_argument('--features', required=False, type=str, default='WMD',
                        help='use "ALL" for all features, or comma separated list of features')
    parser.add_argument('--exclude', required=False, type=str, default='',
                        help='include comma separated list of features to exclude from calculation')
    parser.add_argument('--max_n', type=int, default=1,
                        help='maximum number of n-gram overlap score to calculate, e.g. max_n=2 creates 1-gram-overlap & 2-gram-overlap')
    parser.add_argument('--keep-stopwords', dest='keep_stopwords', action='store_true')
    parser.add_argument('--no-keep-stopwords', dest='keep_stopwords', action='store_false')
    parser.set_defaults(keep_stopwords=True)
    args = parser.parse_args()
    main(args)
