import pandas as pd
from src.features import Metric
from tqdm import tqdm


class NgramOverlap(Metric):

    def __init__(self, n, val='text_'):
        super(NgramOverlap, self).__init__(val=val)
        self.n = n

    @staticmethod
    def gram_overlap(sent_a, sent_b):
        first_sentence_set = set(sent_a)
        second_sentence_set = set(sent_b)
        score_wo = len(first_sentence_set & second_sentence_set) / len(first_sentence_set | second_sentence_set)
        return score_wo

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        metric_names = ['1-gram_overlap']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip().str.split()
        pairs[self.text2] = pairs[self.text2].str.strip().str.split()
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: self.gram_overlap(row[self.text1], row[self.text2]),
                                                      axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
