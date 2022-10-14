"""
run glove & fasttext embedding cosine similarity feature extraction

models:
- glove uses torchtext, and predownloaded can be determined using 'vectors_cache'
- fasttext uses gensim downloader, path is always ~/gensim-data. To control it, make a symlink
"""
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torchtext.vocab as torch_vocab
from src.features import Metric
from tqdm import tqdm


class CosineSimilarity(Metric):

    def __init__(self, val, glove_path=None):
        super(CosineSimilarity, self).__init__(val)
        self.downloaded = False
        self.glove_path = glove_path
        self.models = {}
        self.vocab = {}

    def download(self):
        self.models = dict(glove=torch_vocab.GloVe(name='twitter.27B', dim=100, cache=self.glove_path),
                           fasttext=torch_vocab.FastText(language='en', cache=self.glove_path))
        self.vocab = dict(glove=self.models['glove'], fasttext=self.models['fasttext'])
        self.downloaded = True

    def compute_cs(self, reference: List[str], candidate: List[str], model: str):
        reference_vectors = self.models[model].get_vecs_by_tokens(reference, lower_case_backup=True).numpy()
        candidate_vectors = self.models[model].get_vecs_by_tokens(candidate, lower_case_backup=True).numpy()

        min_reference_vector = np.min(reference_vectors, axis=0)
        min_candidate_vector = np.min(candidate_vectors, axis=0)

        mean_reference_vector = np.mean(reference_vectors, axis=0)
        mean_candidate_vector = np.mean(candidate_vectors, axis=0)

        max_reference_vector = np.max(reference_vectors, axis=0)
        max_candidate_vector = np.max(candidate_vectors, axis=0)

        reference_vector = np.concatenate((min_reference_vector, mean_reference_vector, max_reference_vector))
        reference_vector = reference_vector / np.linalg.norm(reference_vector)
        reference_vector = np.expand_dims(reference_vector, axis=0)

        candidate_vector = np.concatenate((min_candidate_vector, mean_candidate_vector, max_candidate_vector))
        candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)
        candidate_vector = np.expand_dims(candidate_vector, axis=0)

        if np.isnan(reference_vector).any() or np.isnan(candidate_vector).any():
            return None
        else:
            return 1 - cosine_similarity(reference_vector, candidate_vector).item()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.downloaded:
            self.download()        

        print("cosine_similarites start")
        metric_names = ['glove_cosine', 'fasttext_cosine']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip().str.split()
        pairs[self.text2] = pairs[self.text2].str.strip().str.split()
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: self.compute_cs(row[self.text1], row[self.text2],
                                                                                  'glove'), axis=1)
        tqdm.pandas(desc=metric_names[1])
        pairs[metric_names[1]] = pairs.progress_apply(lambda row: self.compute_cs(row[self.text1], row[self.text2],
                                                                                  'fasttext'), axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
