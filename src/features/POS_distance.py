"""implements POS distance metric as a class"""
import pandas as pd
from nltk import pos_tag, word_tokenize
import nltk
import torchtext.vocab as torch_vocab
import torch
from src.features import Metric
from tqdm import tqdm


class POSDistance(Metric):

    def __init__(self, val, vector_path=None):
        super(POSDistance, self).__init__(val=val)
        self.downloaded = False
        self.vector_path = vector_path
        self.dic_glove = None
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
    
    def download(self):
        self.dic_glove = torch_vocab.GloVe(name='twitter.27B', dim=100, cache=self.vector_path)
        self.downloaded = True

    def pos_distance(self, row):
        temp_res_ori = pos_tag(row[self.text1])
        temp_res_gen = pos_tag(row[self.text2])
        temp_nn_ori = []
        temp_nn_gen = []
        temp_nn_vector_ori = []
        temp_nn_vector_gen = []
        for tube in temp_res_ori:
            if tube[1] == 'NN' or tube[1] == 'NNS':
                temp_nn_ori.append(tube[0])
        for tube in temp_res_gen:
            if tube[1] == 'NN' or tube[1] == 'NNS':
                temp_nn_gen.append(tube[0])
        for word in temp_nn_ori:
            try:
                temp_nn_vector_ori.append(self.dic_glove.vectors[self.dic_glove.stoi[word]])
            except KeyError:
                a = 1
        for word in temp_nn_gen:
            try:
                temp_nn_vector_gen.append(self.dic_glove.vectors[self.dic_glove.stoi[word]])
            except KeyError:
                a = 1
        if temp_nn_vector_ori != [] and temp_nn_vector_gen != []:
            loss_list = []
            for vector_target in temp_nn_vector_ori:
                for vector_gen in temp_nn_vector_gen:
                    tensor_gen = torch.FloatTensor(vector_gen)
                    tensor_target = torch.FloatTensor(vector_target)
                    temp_loss = torch.dist(tensor_gen, tensor_target)
                    loss_list.append(temp_loss)
            loss_list_new = sorted(loss_list)
            loss_list_new1 = loss_list_new[:min(len(temp_nn_vector_ori), len(temp_nn_vector_gen))]
            loss = (sum(loss_list_new1) / len(loss_list_new1)) * (
                    1 + abs(len(temp_nn_vector_ori) - len(temp_nn_vector_gen)) / len(temp_nn_vector_ori))
            return loss.numpy().item()
        else:
            return -1

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.downloaded:
            self.download()
        metric_names = ['POS Dist score']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip().str.split()
        pairs[self.text2] = pairs[self.text2].str.strip().str.split()
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: self.pos_distance(row), axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df

