import pandas as pd
import nltk


class Metric:
    def __init__(self, val, keep_stopwords=False):
        self.txt_col_format = val
        self.text2 = f'{self.txt_col_format}2'
        self.text1 = f'{self.txt_col_format}1'
        self.keep_stopwords = keep_stopwords
        self.stopwords = nltk.corpus.stopwords.words('english')
        stopwords_to_remove = ['against', 'no', 'nor', 'not']
        self.stopwords = [word for word in self.stopwords if word not in stopwords_to_remove]

    @staticmethod
    def validate_columns(df, metric_names):
        try:
            df.drop(columns=metric_names, inplace=True)
        except KeyError:
            pass

    def remove_stopwords(self, text_series: pd.Series) -> pd.Series:
        no_stopwords = text_series.apply(lambda text: ' '.join([word for word in text.split() if word not in self.stopwords])).str.strip()
        return no_stopwords

