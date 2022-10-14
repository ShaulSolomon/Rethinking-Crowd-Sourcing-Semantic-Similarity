import pandas as pd
from bert_score import score
from src.features import Metric


class BertScore(Metric):

    def __init__(self, val):
        super(BertScore, self).__init__(val)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # apply function to all word pairs in dataset
        text1 = df[self.text1].str.strip().tolist()
        text2 = df[self.text2].str.strip().tolist()
        _, _, F1 = score(text1, text2, lang='en', verbose=True)
        df['BertScore'] = F1.numpy()
        return df
