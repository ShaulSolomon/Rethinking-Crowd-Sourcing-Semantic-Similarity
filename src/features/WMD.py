"""
calculating wmd using gensim. 
glove zip => txt => gensim => wmdistance

zip => txt => glove2word2vec (gensim) => model.wmdistance
"""
import pandas as pd
import chakin
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import zipfile
from src.features import Metric
from pathlib import Path
import os
from tqdm import tqdm


class WMD(Metric):

    def __init__(self, val, vector_path=None):
        super(WMD, self).__init__(val)
        self.downloaded = False
        self.vector_path = vector_path
        self.model = None
        self.w2vfile_path = os.path.join(self.vector_path, 'glove.840B.300d.txt')
        self.glove_w2v_format = os.path.join(self.vector_path, 'glove.840B.300d.w2v.txt')
        self.zip_path = os.path.join(self.vector_path, 'glove.840B.300d.zip')

    def convert_to_w2v(self):
        print("[WMD] glove=>w2v")
        # glove_file = self.w2vfile_path #datapath(self.w2vfile_path)
        # glove_w2v_format = get_tmpfile(self.glove_w2v_foWrmat)
        _ = glove2word2vec(self.w2vfile_path, self.glove_w2v_format)
        # Path(self.w2vfile_path).unlink(missing_ok=True)

    def download_vectors(self):
        print("[WMD] downloading glove")
        chakin.download(number=16, save_dir=self.vector_path)  # select GloVe.840B.300d

    def unzip_vectors(self):
        print("[WMD] unzipping")
        zip_ref = zipfile.ZipFile(self.zip_path)
        zip_ref.extractall(self.vector_path)
        zip_ref.close()
        Path(self.zip_path).unlink(missing_ok=True)

    def download(self):
        if not os.path.exists(self.glove_w2v_format):
            if not os.path.exists(self.w2vfile_path):
                if not os.path.exists(self.zip_path):
                    self.download_vectors()
                self.unzip_vectors()
            self.convert_to_w2v()
        self.downloaded = True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.downloaded:
            self.download()

        print("[WMD] load model")
        self.model = KeyedVectors.load_word2vec_format(self.glove_w2v_format)
        print("[WMD] model loaded")
        metric_names = ['WMD']
        self.validate_columns(df, metric_names)
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].str.strip()
        pairs[self.text2] = pairs[self.text2].str.strip()

        print("[WMD] after update pairs")
        tqdm.pandas(desc=metric_names[0])
        pairs[metric_names[0]] = pairs.progress_apply(lambda x: self.model.wmdistance(x[self.text1], x[self.text2]),
                                                      axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
