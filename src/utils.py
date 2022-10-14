import pandas as pd
import numpy as np
from pathlib import Path
import configparser
import wandb
import sys

MODULE_PATH = Path(__file__).parents[1].resolve()
sys.path.insert(0,str(MODULE_PATH))

from src import model_corr
from src import metric_exploration
from src.testing import df_validation
import wandb
from typing import Union
import matplotlib.pyplot as plt

FILEPATH = Path(Path.cwd().parents[1].resolve()) / 'data'
CRED_PATH = Path(__file__).resolve().parents[1] / 'credentials.ini'


path_combined = 'combined_dataset.csv'
path_sts = 'sts.csv'
path_qqp = 'qqp.csv'
path_qqp_sample = 'sample_qqp.csv'
path_ba = 'ba_all.txt'

METRICS = ['bleu', 
           'bleu1',
           'glove_cosine',
           'fasttext_cosine',
           'BertScore',
           'chrfScore',
           'POS Dist score',
           '1-gram_overlap',
           'ROUGE-1',
           'ROUGE-2',
           'ROUGE-l',
           'L2_score',
           'WMD']

DISTANCE_METRICS = ['glove_cosine',
                    'fasttext_cosine',
                    'BertScore',
                    'POS_Dist_score',
                    'L2_score',
                    'WMD']

SELECTED_METRICS = ['WMD','BertScore', 'POS Dist score']


def get_environment_variables():
    config_parser = configparser.ConfigParser()
    config_parser.read(CRED_PATH)
    PATH_ROOT = config_parser['MAIN']["PATH_ROOT"]
    PATH_DATA = config_parser['MAIN']["PATH_DATA"]
    GloVe_840B_300d_PATH = config_parser['MAIN']["GloVe_840B_300d_PATH"]
    Glove_twitter_27B_PATH = config_parser['MAIN']["Glove_twitter_27B_PATH"]
    ENV = config_parser['MAIN']["ENV"]
    return PATH_ROOT, PATH_DATA, GloVe_840B_300d_PATH, Glove_twitter_27B_PATH, ENV


def convert_tsv_to_csv_sts(path=FILEPATH):
    """
    Load STS benchmark data. - from the Github sourcecode
    """
    genres, sent1, sent2, labels, scores = [], [], [], [], []
    for line in open(path):
        genre = line.split('\t')[0].strip()
        filename = line.split('\t')[1].strip()
        year = line.split('\t')[2].strip()
        other = line.split('\t')[3].strip()
        score = line.split('\t')[4].strip()
        s1 = line.split('\t')[5].strip()
        s2 = line.split('\t')[6].strip()
        label = float(score)
        genres.append(genre)
        sent1.append(s1)
        sent2.append(s2)
        labels.append(label)
        scores.append(score)
    labels = (np.asarray(labels)).flatten()

    # include which category (train/dev/test) as column for future reference
    dataset_categ = [os.path.split(path)[1].replace(".csv", "")] * len(labels)
    dataset = ['sts'] * len(labels)

    return pd.DataFrame({'genres': genres, 'text_1': sent1, 'text_2': sent2,
                         'label': labels, 'scores': scores, 'dataset': dataset, 'dataset-categ': dataset_categ})


class Config():
    '''
    Configuration of the parameters to test base_metric and RF_correlation score.
        train_dataset -- {Path} -- path of the train dataset
        test_dataset -- {Union[Path, None]} -- path of the test dataset. If None - the data will be split 80/20 between train/test
                                            slight modification for sts as internally is split train/dev/test
        bad_annotators -- {Union[Path,None]} -- path of the bad_annotator list we wish to take. Only used in case of combined_dataset
        scale_features -- {bool} -- Min/Max scaling and converting all to similarity metrices
        scale_labels -- {bool} -- Turns [0,5] to {-1,0,1} labeling
        rf_depth  -- {int} -- Depth of the Random Forest Trees
        rf_top_n_features -- {Union[int,None]} -- the number of features to take. If None - will take all the features
        metrics -- {list} -- the list of metrics to explore correlation on. By default we take all the metrics.
    '''

    def __init__(self, 
                 train_dataset: Path,
                 test_dataset: Union[Path,None] = None,
                 bad_annotators: Union[Path,None] = None,
                 scale_features: bool = True,
                 scale_labels: bool = False,
                 rf_depth: int = 6,
                 rf_top_n_features: int = 3,
                 metrics = METRICS
                 ):
        self.config = dict()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.bad_annotators = bad_annotators
        self.scale_features = scale_features
        self.scale_labels = scale_labels
        self.rf_depth = rf_depth
        self.rf_top_n_features = rf_top_n_features
        self.metrics = metrics
    
    def __getitem__(self, key):
        return self.__dict__[key]


def wandb_logging(config, project_name = "semantic_similarity", run_wandb=True):
    '''
    Call src.model_corr to explore the base_metrics and and RF correlation score and to save the important details to wandb.

    Parameters:
        config -- {Config} -- instance of the Config class with the parameters determined
        project_name -- {str} -- name of the project the trial will be saved within Wandb
        run_wandb -- {bool} -- Default is true. Used in false if want to debug code and dont want to send to wandb.
    '''


    if run_wandb:
        run = wandb.init(project=project_name,config=config,reinit=True)

    X_train, X_test, y_train, y_test = model_corr.get_train_test_data(train_path = config.train_dataset, 
                                                                      test_path = config.test_dataset,
                                                                      all_metrics = config.metrics,
                                                                      filtered_ba_path = config.bad_annotators,
                                                                      scale_features = config.scale_features,
                                                                      scale_label = config.scale_labels)

    base_metrics = X_test.corrwith(y_test).apply(lambda x: abs(x)).sort_values(ascending=False).reset_index()

    if run_wandb:
        wandb.log({"Base Top Correlation": base_metrics.iloc[0][0]})

    base_metrics.columns = ['Features','Importance']

    if run_wandb:
        table = wandb.Table(dataframe=base_metrics, columns=["Features","Importance"])
        wandb.log({"Base Metrics": wandb.plot.bar(table, "Features", "Importance", title="Base Metric Table")})
    
    pearsonr, features = model_corr.RF_corr(X_train,X_test,y_train,y_test,config.rf_depth, config.rf_top_n_features)
    features.columns = ["Features", "Importance"]

    if run_wandb:
        table2 =  wandb.Table(dataframe=features, columns=["Features","Importance"])
        wandb.log({"RF PearsonR": pearsonr, "RF Metrics":  wandb.plot.bar(table2, "Features", "Importance", title="RF Metric Table")})
        run.finish()
    else:
        return base_metrics, pearsonr, features


if __name__ == "__main__":
    pass
    