import pandas as pd
import os
import numpy as np
import pickle
import glob

for i, file in enumerate(glob.glob('data/raw_data/*.csv')):
    if i==0:
        df = pd.read_csv(file, index_col = 0)
        df['dataset'] = file[5:-4].lower()
    else:
        temp = pd.read_csv(file, index_col = 0)
        temp['dataset'] = file[5:-4].lower()
        df = pd.concat((df,temp),axis = 0)
df['random'] = df.dataset.apply(lambda x: 'random' in  x).astype(int)
df['duration'] = pd.to_datetime(df.SubmitTime)-pd.to_datetime(df.AcceptTime)

df = df.drop(['HITId','HITTypeId', 'Title', 'Description', 'Keywords',
       'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AssignmentId','AssignmentStatus','AcceptTime', 'SubmitTime','AutoApprovalTime', 'ApprovalTime', 'RejectionTime','RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate','Last30DaysApprovalRate', 'Last7DaysApprovalRate','Approve', 'Reject'],axis = 1)

df=df.rename(columns = {'Input.text1':'text_1','Input.text2':'text_2','Answer.semantic-similarity.label':'label'})


df.label = df.label.apply(lambda x:x[0]).astype(int)
df = df.reset_index()
df.drop(columns="index",inplace=True)
df['total_seconds'] = [int(df['duration'][i].total_seconds()) for i in range(df.shape[0])]

#As there is no id for same pair documents - I will create it
df["pair_id"] = [f"pair_{i//3}" for i in range(df.shape[0])]

#We will first replace 1-2 with [-1] and 4-5 with [1]
df['reduced_label'] = [1 if x > 3 else -1 if x < 3 else 0 for x in df.label]

df.to_csv('data/combined/combined_dataset.csv')