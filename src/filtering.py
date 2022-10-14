import pandas as pd
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm
tqdm.pandas()


class filter_annotators:
    def __init__(self, labelers_col = 'annotator', random_col = 'random', duration_col = 'total_seconds',
                label_col = 'label', reduced_col = 'reduced_label', 
                text_1_col = 'text_1', text_2_col = 'text_2', sentences_id = 'pair_id',
                time_method = 'seconds', time_limit = 300,
                min_labels = 4, min_var = 1, unpopular_share = 0.5,
                inconsistent_thresh = 1,bleu_thresh = 0.8, sent_thresh = 1.9,
                exclude = ''):
        '''
        Arguments:
            labelrs_col {str}: name of labeler_id column
            random_col {str}: name of column that indicated wheather a pair is randomly assigned or not
            duration_col {str}: name of columns with labeling length indication
            reducel_col {str}: name of column with labels scaled between -1 and 1
            text_1_col {str}: name of column with the first sentence
            text_2_col {str}: name of column with the second sentence
            sentences_id {str}: name of column with identifyer for sentence pair
            time_method {str}: argument for time outlier function, wheather to use seconds or percentiles
            time_limit {float or int}: argument for time outlier function, high bound cap
            min_labels {int}: argument for varience and honeypot functions, minimum number of labels for labelers to calcukate groupby's
            min_var {float or int}: argument for varience function, minimal threshold value
            unpopular_share {float}: argument for unpopular voter function, threshold value of the share of labels being on the monirity vote
            inconsistent_thresh {float or int}: argument for inconsistenet_sentiment function , low bound for inconsistency measurement 
            bleu_thresh {float}: argument for inconsistenet_sentiment function , low bound for Bleu similarity score between sentences
            sent_thresh {float}: argument for inconsistenet_sentiment function , low bound for sentiment difference between sentences
            exclude {str or list}: functions not to run [time_outliers, random_honeypot,high_random, no_varience, unpopular, sentiment_inconsistent]
        '''
        self.df = None
        self.labelers = None
        self.labelers_col = labelers_col
        self.random_col = random_col
        self.duration_col = duration_col
        self.label_col = label_col
        self.reduced_col = reduced_col
        self.text_1 = text_1_col
        self.text_2 = text_2_col
        self.time_method = time_method
        self.time_limit = time_limit
        self.sentences_id = sentences_id
        self.min_labels = min_labels
        self.min_var = min_var
        self.unpopular_share = unpopular_share
        self.inconsistent_thresh = inconsistent_thresh
        self.bleu_thresh = bleu_thresh
        self.sent_thresh = sent_thresh        
        if isinstance(exclude, str):
            self.exclude = [exclude]
        self.exclude = exclude
        
        self.ba = {'duration': None,
                    'random_honeypot': None,
                    'low_std': None,
                    'high_random': None,
                    'unpopular': None,
                    'sentiment_inconsistent': None,
                    'ba_combined': None}

    def fit (self, df):
        '''
        preforms all the filtering functions listed below:
            - time_outliers
            - random_honeypot
            - high_random
            - no_varience
            - unpopular_voter
            - inconsistenet_sentiment
        stores each list of suspicious labelers in the dictionary and creates a combined list of suspicious labelers
        '''
        labelers = None

        self.df = df
        if self.reduced_col is None:
            self.df['reduced_label'] = self.df[self.label_col].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0) 
            self.reduced_col = 'reduced_label'
       
        # set up random honey pot and no varience filters, 
        # this filter takes arguments that satisfy both conditions: 
        #   - low varience 
        #   - higher random mean than non-random       
        if self.random_col is not None:
            rands = self.random_honey_pot()
            stds = self.no_variance()
            
            if 'random_honeypot' not in self.exclude:
                self.ba['random_honeypot'] = list(set(rands) \
                                        .intersection(set(stds)))
            if 'high_random' not in self.exclude:
                self.ba['high_random'] = rands
            if 'no_varience' not in self.exclude:
                self.ba['low_std'] = stds


        # set up time outliers filter
        if self.duration_col is not None:     
            if 'time_outliers' not in self.exclude:
                self.ba['duration'] = self.time_outliers (method = self.time_method, limit = self.time_limit)
            
        #set up unpopular vote
        if self.sentences_id is not None:
            if 'unpopular' not in self.exclude:
                self.ba['unpopular'] = self.unpopular_voter(self.min_labels,self.unpopular_share)

        #set up inconsistent sentiment
        if 'sentiment_inconsistent' not in self.exclude:
            self.ba['sentiment_inconsistent'] = self.inconsistenet_sentiment(thresh = self.inconsistent_thresh, 
                                                                            bleu_thresh = self.bleu_thresh , sent_thresh = self.sent_thresh)
        
        if labelers is not None:
            self.labelers = labelers

        #combine all the lists together
        combined = []
        for lst in self.ba.values():
            if lst is not None:
                combined += lst
        self.ba['ba_combined'] = list(set(combined))


    def time_outliers(self, method = 'seconds', limit = 300):
        '''
        filter annotators based on the time it took to label a pair of sentences
        argument in:
            - duration series {series, array or list} - time it took to annotate
            - method {str} - weather to filter based on a cap value or cap percentile default is seconds (options: seconds, percentile)
            - limit {int, float or None} - if number: the ceiling value to cap the duartion (default 300), \
                                                    if method = None does it on a percentile based value.
        Returns:
            time outliers list (index)
        
        '''
        if self.labelers is not None:
            self.labelers = self.labelers.join(pd.DataFrame(self.df.groupby(self.labelers_col)[self.duration_col].mean(),columns = [self.duration_col] ))           
        else:
            self.labelers = pd.DataFrame(self.df.groupby(self.labelers_col)[self.duration_col].mean())
    
        if method =='seconds':
            return list(self.labelers[self.labelers[self.duration_col]>limit].index.values)
        else:
            return list(self.labelers[self.labelers[self.duration_col] > self.labelers[self.duration_col].quantile(limit)].index.values)

    def random_honey_pot (self):
        '''
        identify which labelers have a higher mean for random pairs than non-random pairs
        Returns:
            list of suspicious labelers
        '''
        # calculate descriptives for non random pairs
        labelers = self.df[self.df[self.random_col]==0].groupby(self.labelers_col)[self.label_col].agg(['mean','std','size'])
        labelers = labelers[labelers['size']>1]
        # calculate descriptives for non random pairs
        labelers_rand = self.df[self.df[self.random_col]==1].groupby(self.labelers_col)[self.label_col].agg(['size','mean','std'])
        labelers_rand = labelers_rand[labelers_rand['size']>1]
        labelers = labelers.join(labelers_rand, rsuffix = '_rand')
        labelers['mean_random_gap'] = labelers['mean']-labelers['mean_rand']
        if self.labelers is not None:
            self.labelers = self.labelers.join(labelers, rsuffix = '_labels')
        else:
            self.labelers = labelers
        
        return list((self.labelers[self.labelers['mean_random_gap']<0]).index.values)

    def no_variance (self, min_var = 1):
        '''
        Identify which labelers doesn't vary enough in their labels
        Args:
            min_var {float or int} - minimal standard deviation value to compare to.
        return:
            list of suspicious labelers
        '''
        total_std = self.df.groupby(self.labelers_col)[self.label_col].std()
        total_std.name = 'total_std'
        if self.labelers is not None:
            self.labelers = self.labelers.join(total_std)
        else:
            self.labelers = labelers
        return list((self.labelers[self.labelers['total_std']<min_var]).index.values)

    def unpopular_voter(self, min_labels = 4 , unpopular_share = 0.5):
        '''
        Identify which labelers tend to be on the unpopular opinion
        Args:
            min_labels {int} - minimal number of labels for labeler
            unpopular share {float} - threshold share of labels that were on the unpopular size
        Returns:
            list of suspicious labelers
        '''

        # count number of different answers for each sentence pair
        uniquelabels = self.df.groupby(self.sentences_id)[self.reduced_col].nunique()
        # reduce dataset to only have 2 unique answers
        pairs_two_agree = uniquelabels[uniquelabels==2].index.values
        df_twoagree = self.df[self.df[self.sentences_id].isin(pairs_two_agree)]    

        # set the generally agreed labels as popular vote 
        df_id_reducedlabel = df_twoagree.groupby(self.sentences_id)[self.reduced_col].median()
        df_twoagree = df_twoagree.set_index(self.sentences_id).join(df_id_reducedlabel.rename('generally_accepted_label'))
        
        
        #count for each labeler how many times they were on the unpopular vote
        df_unpopularopinion = pd.DataFrame(df_twoagree[df_twoagree.reduced_label != df_twoagree.generally_accepted_label].groupby(self.labelers_col).size(), columns = ['unpopular_opinion_times'])

        # count nummber of labels per labeler
        total_opinion = pd.DataFrame(self.df.groupby(self.labelers_col).size(),columns = ['total_opinion'])
        
        #create df with both columns
        total_opinion = total_opinion.join(df_unpopularopinion).fillna(0)
        total_opinion['unpopular_share'] = total_opinion.unpopular_opinion_times/total_opinion.total_opinion
        
        if self.labelers is not None:
            self.labelers = self.labelers.join(total_opinion['unpopular_share'])
        else:
            self.labelers = labelers

        return list(((total_opinion[(total_opinion.total_opinion>min_labels) & (total_opinion.unpopular_share>unpopular_share)])).index.values)

    def inconsistenet_sentiment(self, thresh = 1, bleu_thresh = 0.8, sent_thresh = 1.9):
        '''
        identify labelers who show inconsistenet labeling stratagy: 
        for sentence pairs which have high BLEU score (i.e most words are similar), but high sentiment difference (i.e probably negation between them)
        the labeler have high std. meaning that the stratagy for labeling similar sentneces with opposite sentiment is inconsistenet
        args:
            thresh {float}
            bleu_thresh {float}
            sent_thresh {float}
        Returns :
            list of suspicious labelers
        '''
        
        
        sent = {'POSITIVE':1,'NEGATIVE':-1}
        
        # set up sentiment analysis pipeline from transformers package https://huggingface.co/transformers/main_classes/pipelines.html
        sentiment_pipe = pipeline("sentiment-analysis")

        # calculate sentiment score for each sentence
        df = self.df.copy()
        df['text_1_sent'] = df[self.text_1].progress_apply(lambda x: sentiment_pipe(x)).apply(lambda x: sent[x[0]['label']]*x[0]['score'])
        df['text_2_sent'] = df[self.text_2].progress_apply(lambda x: sentiment_pipe(x)).apply(lambda x: sent[x[0]['label']]*x[0]['score'])
        df['sent_diff'] = (df['text_1_sent'] - df['text_2_sent']).abs()

        self.df['sent_diff'] = df['sent_diff'] 
        
        #std high bleu high diff
        bleu_diff_std = df[(df['bleu1'] > bleu_thresh) & (df['sent_diff'] > sent_thresh)].groupby(self.labelers_col)[self.label_col].std()

        #filter out annotator who show inconsistent stratagy for sentences with high BLEU but different sentiment 
        return list(bleu_diff_std[bleu_diff_std > thresh].dropna().index)


        
