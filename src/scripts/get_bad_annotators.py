import pandas as pd
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2].resolve()
sys.path.insert(0,str(MODULE_PATH))
from src import filtering

import pickle 
import numpy as  np


def main():
    combined = pd.read_csv(MODULE_PATH /'data'/'datasets'/'combined_dataset.csv', index_col = 0)


    filt_combined = filtering.filter_annotators(text_1_col = 'text1', text_2_col = 'text2')
    filt_combined.fit(combined)
    combined = filt_combined.ba

    with open (MODULE_PATH / 'data'/'bad_annotators'/'combined_ba.pickle', 'wb') as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
