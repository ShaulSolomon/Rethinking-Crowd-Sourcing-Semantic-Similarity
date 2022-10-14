"""
run bleu feature extract
"""
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from src.features import Metric
from tqdm import tqdm


class Bleu(Metric):

    def __init__(self, val, stopwords=False):
        super(Bleu, self).__init__(val=val, keep_stopwords=stopwords)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        metric_names = ['bleu', 'bleu1']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        if self.keep_stopwords:
            pairs[self.text1] = self.remove_stopwords(pairs[self.text1]).str.strip().str.split()
            pairs[self.text2] = self.remove_stopwords(pairs[self.text2]).str.strip().str.split()
        else:
            pairs[self.text1] = pairs[self.text1].str.strip().str.split()
            pairs[self.text2] = pairs[self.text2].str.strip().str.split()
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: sentence_bleu([row[self.text1]], row[self.text2]),
                                                      axis=1)
        tqdm.pandas(desc=metric_names[1])
        pairs[metric_names[1]] = pairs.progress_apply(lambda row: sentence_bleu([row[self.text1]], row[self.text2],
                                                                                weights=(1, 0, 0, 0)), axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df

