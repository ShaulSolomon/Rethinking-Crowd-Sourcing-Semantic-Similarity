from typing import List, Union
import pandas as pd
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
import torch
from src.features import Metric
from tqdm import tqdm


class EuclideanElmoDistance(Metric):

    def __init__(self, val):
        super(EuclideanElmoDistance, self).__init__(val=val)
        self.downloaded = False
        self.embeddings = None

    def download(self):        
        self.embeddings = ELMoEmbeddings()
        self.downloaded = True

    def create_embedding(self, sentence: str) -> torch.Tensor:

        if not self.downloaded:
            self.download()
                    
        # embed words in sentence
        sent = Sentence(sentence)
        self.embeddings.embed(sent)
        # return average embedding of words in sentence
        return torch.stack([token.embedding for token in sent]).mean(dim=0)

    def l2_distance(self, candidate: str, reference: str) -> Union[float, None]:
        candidate_embedding = self.create_embedding(candidate)
        reference_embedding = self.create_embedding(reference)
        if candidate_embedding.shape[0] != reference_embedding.shape[0]:
            return None
        else:
            return torch.norm(candidate_embedding - reference_embedding, p=2).item()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        metric_names = ['L2_score']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip()
        pairs[self.text2] = pairs[self.text2].str.strip()
        print('Calculating Elmo L2 distance')
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: self.l2_distance(row[self.text1], row[self.text2]),
                                                      axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
