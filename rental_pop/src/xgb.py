import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

TOLERANCE = 20
EPOCHS = 5000
TEST_RATIO = 0.2 #must sum to 1

#input data
train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')

#basic features
train_df["price_t"] = train_df["price"]/train_df["bedrooms"]
test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# cross variables
abc_list = []
for i in xrange(97, 123):
    abc_list.append(str(chr(i)))
train_lon, lon_bins = pd.qcut(train_df["longitude"], 10, retbins=True, labels=abc_list[0:10])
train_lat, lat_bins = pd.qcut(train_df["latitude"], 10, retbins=True, labels=abc_list[0:10])
train_lon = train_lon.astype(object)
train_lat = train_lat.astype(object)
train_df["grid"] = train_lon + train_lat

test_lon = pd.cut(test_df["longitude"], lon_bins, labels=abc_list[0:10]).astype(object)
test_lat = pd.cut(test_df["latitude"], lat_bins, labels=abc_list[0:10]).astype(object)
test_df["grid"] = test_lon + test_lat

print('End of initial feature engineering')

features_to_use=["bathrooms", "bedrooms", "longitude", "latitude",
                 "price","price_t","num_photos", "num_features", "num_description_words",
                 "listing_id"]

print("Start of categorical feature engineering")
categorical = ["display_address", "manager_id", "building_id", "street_address", "grid"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            # train_one = train_df[f].astype(object).reshape(-1, 1)
            # test_one = train_df[f].astype(object).reshape(-1, 1)
            # one_hot = OneHotEncoder()
            # one_hot.fit(train_one)
            # train_df[f] = one_hot.transform(train_one)
            # test_df[f] = one_hot.transform(test_one)
            features_to_use.append(f)

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=EPOCHS):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=TOLERANCE)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

rows = len(train_df)
# get number of row and create randomized indexs
index=list(range(rows))
random.shuffle(index)
# initialize new feature lists
a2=[np.nan]*rows
b2=[np.nan]*rows
c2=[np.nan]*rows

print('Training new features for training dataset')
batches = int(1/TEST_RATIO)

for i in range(batches):
    print('Test ' + str(i + 1) + ' of ' + str(batches))
    grid_level={}
    # loop through manager_ids and initialize by unique ids
    for j in train_df['grid'].values:
        grid_level[j]=[0,0,0]
    # split indexes into 5 but avoided hotspot with shuffle above
    test_index=index[int((i*rows)/batches):int(((i+1)*rows)/batches)]
    # get indexes not in test_index but in index i.e. the rest
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        # select listing by index
        temp=train_df.iloc[j]
        # tally interest level by manager_id
        if temp['interest_level']=='low':
            grid_level[temp['grid']][0]+=1
        if temp['interest_level']=='medium':
            grid_level[temp['grid']][1]+=1
        if temp['interest_level']=='high':
            grid_level[temp['grid']][2]+=1

    for j in test_index:
        temp=train_df.iloc[j]
        gcount = sum(grid_level[temp['grid']])
        if gcount !=0:
            a2[j]=grid_level[temp['grid']][0]*1.0/gcount
            b2[j]=grid_level[temp['grid']][1]*1.0/gcount
            c2[j]=grid_level[temp['grid']][2]*1.0/gcount

# after looping through all rows, create the features
train_df['grid_level_low']=a2
train_df['grid_level_medium']=b2
train_df['grid_level_high']=c2

print('Creating new features for test data set')

a2=[]
b2=[]
c2=[]

grid_level={}

# still using training data manager IDs
for j in train_df['grid'].values:
    grid_level[j]=[0,0,0]

# use all training data and tally manager scores
for j in range(rows):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        grid_level[temp['grid']][0]+=1
    if temp['interest_level']=='medium':
        grid_level[temp['grid']][1]+=1
    if temp['interest_level']=='high':
        grid_level[temp['grid']][2]+=1

# finally we play with the test dataset
for i in test_df['grid'].values:
    # for managers with no levels
    if i not in grid_level.keys():
        a2.append(np.nan)
        b2.append(np.nan)
        c2.append(np.nan)
    else:
        a2.append(grid_level[i][0]*1.0/sum(grid_level[i]))
        b2.append(grid_level[i][1]*1.0/sum(grid_level[i]))
        c2.append(grid_level[i][2]*1.0/sum(grid_level[i]))

test_df['grid_level_low']=a2
test_df['grid_level_medium']=b2
test_df['grid_level_high']=c2

features_to_use.append('grid_level_high')
features_to_use.append('grid_level_high')
features_to_use.append('grid_level_high')

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])
    
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break

print('final prediction')
preds, model = runXGB(train_X, train_y, test_X, num_rounds=EPOCHS)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("../output/xgb_2.csv", index=False)
