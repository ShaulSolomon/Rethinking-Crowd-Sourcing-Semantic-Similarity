import unittest
from src.run_feature_extract import main


class Config:
    """Config class for debugging in IDE"""

    def __init__(self, pckl: str, features: str = 'ALL', max_n: int = 1, exclude: str = None):
        self.pickle = pckl
        self.features = features
        self.max_n = max_n
        self.exclude = exclude
        self.keep_stopwords = True


feats = 'rouge'
exclusion = 'wmd,elmo,bert'
datapath = '/Users/adam/SOTA/data/datasets/combined_dataset.csv'
arguments = Config(datapath, feats, 1, exclusion)
main(arguments)


# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()
