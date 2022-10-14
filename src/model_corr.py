import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

metrics = ['bleu', 
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

distance_metrics = ['glove_cosine',
                    'fasttext_cosine',
                    'BertScore',
                    'POS_Dist_score',
                    'L2_score',
                    'WMD']

MODULE_PATH = Path(__file__).resolve().parents[1].resolve()
DATA_PATH = MODULE_PATH / 'data' / 'datasets'
BA_PATH = MODULE_PATH / 'data' / 'bad_annotators'


def get_train_test_data(train_path: str, all_metrics: list = metrics, test_path: str = None, filtered_ba_path = None, scale_features = True, scale_label=False) -> tuple:
    '''
    Retrieve the data from the paths and split into train/test based off if they are from the same dataset or otherwise.

    Parameters:
        train_path -- {str} -- Path to the training dataset
        metrics -- {list} -- name of the metrics we want to use as features
        test_path -- {str} -- Path to the test dataset (by default is None, and then is the same as Train_path)
        filtered_ba -- {list} -- list of annotators to filter out

    Returns:
        {tuple} -- (X_train, X_test, y_train, y_test)
    '''
    if ("combined_dataset.csv" != train_path) and ("combined_dataset.csv" != test_path):
        assert (filtered_ba_path is None), "Filtering only accesible in combined_dataset"

    
    if filtered_ba_path:
        with open(BA_PATH / filtered_ba_path) as f:
            filtered_ba = f.read().split("\n")

 
    if test_path is None:
        df = pd.read_csv(DATA_PATH / train_path, index_col=0)

        #drop null values
        df.dropna(inplace=True)

        if filtered_ba_path:
            df = df[~df.annotator.isin(filtered_ba)]

        if scale_features:
            df, _ = scale_for_similarity(df)

        if scale_label:
            if max(df.label) != 1:
                df['label'] = [1 if score > 3 else -1 if score < 3 else 0 for score in df.label]
            else:
                df['label'] = [1 if score == 1 else -1 for score in df.label]
        
        #If we are dealing with the sts dataset, where it has within it a pre-defined train/val/test
        if Path(train_path).stem == 'sts':
            train_data = df[df['dataset-categ'] == 'sts-train']
            #to include both 'sts-dev' and 'sts-test'
            test_data  = df[df['dataset-categ'] != 'sts-train']
        else:
            #shuffle the dataframe
            len_df = int(df.shape[0] * 0.8)

            df = df.sample(frac=1, random_state=42)
            train_data= df.iloc[:len_df]
            test_data = df.iloc[len_df:]
    else:
        train_data = pd.read_csv(DATA_PATH / train_path, index_col=0)
        test_data = pd.read_csv(DATA_PATH / test_path, index_col=0)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        if filtered_ba_path:
            if train_path == "combined_dataset.csv":
                train_data = train_data[~train_data.annotator.isin(filtered_ba)]
            else:
                test_data = test_data[~test_data.annotator.isin(filtered_ba)]


        if scale_features:
            train_data, test_data = scale_for_similarity(train_data, test_data)

        if scale_label:
            if max(train_data.label) != 1:
                train_data['label'] = [1 if score > 3 else -1 if score < 3 else 0 for score in train_data.label]
            else:
                train_data['label']= [ 1 if score == 1 else -1 for score in train_data.label]
            
            if max(test_data.label) != 1:
                test_data['label'] = [1 if score > 3 else -1 if score < 3 else 0 for score in test_data.label]
            else:
                test_data['label'] = [ 1 if score == 1 else -1 for score in test_data.label]

    #To test it on 
    metrics = [x for x in test_data.columns if x in all_metrics]
    if len(metrics) != len(all_metrics):
        print(f"Still missing the following metrics: {set(all_metrics).difference(set(metrics))}")

    print(f"Size of train_data: {train_data.shape[0]}\tSize of test_data: {test_data.shape[0]}")
    return (train_data[metrics], test_data[metrics], train_data['label'], test_data['label'])

def scale_for_similarity(train_df, test_df = None):
    scaler = MinMaxScaler()

    for column in metrics:
        train_df[column] = scaler.fit_transform(train_df[column].values.reshape(-1,1))
        if test_df is not None:
            test_df[column] = scaler.transform(test_df[column].values.reshape(-1,1))

        if column in distance_metrics:
            train_df[column] = 1 - train_df[column]
            if test_df is not None:
                test_df[column] = 1 - test_df[column]

    return train_df, test_df


####### RF #######

def RF_corr(X_train,X_test,y_train,y_test, max_depth = 6, top_n_features = None):
    '''
    Random Forest Regression.

    Parameters:
        max_depth -- {int} -- depth of the Random Forest Regressor
        X_train -- {pd.DataFrame} -- Train data
        y_train -- {pd.Series} -- Train labels

    Return:
        y_pred -- {list} -- Test predicted labels
        model -- {model} -- The RF Model

    '''
    model = RandomForestRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    features = pd.DataFrame([model.feature_importances_], columns=X_train.columns, index = ["Importance"]).T
    features = features.sort_values("Importance",ascending=False)


    if top_n_features:
        top_features = features[:top_n_features].index
        model.fit(X_train[top_features], y_train)
        y_pred = model.predict(X_test[top_features])
    else:
        y_pred = model.predict(X_test)

    return pearsonr(y_pred,y_test)[0], features.reset_index()


##################

###  MLP MODEL ###

class DS(Dataset):
    '''
    Basic Dataset for the MLP.
    '''
    def __init__(self,df,labels):
        super(DS).__init__()
        self.df = df
        self.labels = labels

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        feat = self.df[idx,:]
        label = self.labels[idx]        

        return feat,label

class Basemodel(nn.Module):
  
  def __init__(self,n_feature,n_hidden,n_output, keep_probab = 0.1):
    '''
    input : tensor of dimensions (batch_size*n_feature)
    output: tensor of dimension (batchsize*1)
    '''
    super().__init__()
  
    self.input_dim = n_feature    
    self.hidden = nn.Linear(n_feature, n_hidden)
    self.hidden2 = nn.Linear(n_hidden, n_hidden // 2) 
    self.predict = nn.Linear(n_hidden // 2, n_output)
    self.dropout = nn.Dropout(keep_probab)
    self.dropout2 = nn.Dropout(keep_probab)
    # self.pool = nn.MaxPool2d(2, 2)
    # self.norm = nn.BatchNorm2d(self.num_filters


  def forward(self, x):
    x = self.dropout(F.relu(self.hidden(x)))
    x = self.dropout2(F.relu(self.hidden2(x)))
    x = self.predict(x)
    return x

def train_epochs(tr_loader,model,criterion,optimizer, num_epochs):

    log_loss = []

    if torch.cuda.is_available():
      device = torch.device('cuda:0')
      model.to(device)
    else:
      device = torch.device('cpu:0')

    for epoch in range(num_epochs):
    #   print("started training epoch no. {}".format(epoch+1))
        epoch_loss = 0
        for step,batch in enumerate(tr_loader):
            feats,labels = batch
            feats = feats.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        log_loss.append(epoch_loss / tr_loader.__len__())

    plt.plot(list(range(num_epochs)), log_loss)
    return model

def MLP_corr(X_train,X_test,y_train,y_test, num_hl = 128, batch_size = 128, num_epochs=100, lr = 1e-3):
    model = Basemodel(X_train.shape[1],num_hl,1)
    criterion = nn.MAELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = torch.Tensor(X_test.to_numpy()).to(dtype=torch.float32)

    train_set = DS(X_train,y_train)
    train_loader=DataLoader(dataset= train_set, batch_size = batch_size, shuffle = True, num_workers = 2)

    model = train_epochs(train_loader,model,criterion,optimizer,num_epochs= num_epochs)
    
    if torch.cuda.is_available:
        y_pred = model(X_test).cpu().detach().numpy().flatten()
    else:
        y_pred = model(X_test).detach().numpy().flatten()

    return pearsonr(list(y_pred),list(y_test))

##################