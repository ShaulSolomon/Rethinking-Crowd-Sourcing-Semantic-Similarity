import pandas as pd
from nltk.translate import chrf_score
from src.features import Metric
from tqdm import tqdm


class chrFScore(Metric):

    def __init__(self, val):
        super(chrFScore, self).__init__(val=val)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        metric_names = ['chrfScore']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip().str.split()
        pairs[self.text2] = pairs[self.text2].str.strip().str.split()
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: chrf_score.sentence_chrf(row[self.text1],
                                                                                           row[self.text2]), axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
