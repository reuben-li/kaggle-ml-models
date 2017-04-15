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
from sklearn import tree
from Levenshtein import distance

TOLERANCE = 30
EPOCHS = 5000

# input data
train_df=pd.read_json('../input/train.json', convert_dates=["created"])
test_df=pd.read_json('../input/test.json', convert_dates=["created"])

# basic features
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

# difference between addresses
train_df["address_distance"] = train_df[["street_address", "display_address"]].apply(lambda x: distance(*x), axis=1)
test_df["address_distance"] = test_df[["street_address", "display_address"]].apply(lambda x: distance(*x), axis=1)

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

le = preprocessing.LabelEncoder()
le.fit(train_df["grid"].append(test_df["grid"]))
train_df["grid"] = le.transform(train_df["grid"]) 
test_df["grid"] = le.transform(test_df["grid"]) 

print('predicting price profile')
clf = tree.DecisionTreeClassifier()
params = ['bedrooms', 'bathrooms', 'num_features', 'grid']
clf = clf.fit(train_df[params], train_df['price'])
train_df["exp_price"] = pd.DataFrame(clf.predict(train_df[params]).tolist()).set_index(train_df.index)
train_df["overprice"] = train_df["price"] - train_df["exp_price"]

test_df["exp_price"] = pd.DataFrame(clf.predict(test_df[params]).tolist()).set_index(test_df.index)
test_df["overprice"] = test_df["price"] - test_df["exp_price"]

print([train_df.iloc(10)])

print('End of feature engineering')

features_to_use=["latitude", "longitude", "bathrooms", "bedrooms", "address_distance",
                 "price","price_t","num_photos", "num_features", "num_description_words",
                 "listing_id", "overprice", "exp_price"]

categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
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
    
index=list(range(train_df.shape[0]))
random.seed(0)
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    manager_level={}
    for j in train_df['manager_id'].values:
        manager_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            manager_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            manager_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(manager_level[temp['manager_id']])!=0:
            a[j]=manager_level[temp['manager_id']][0]*1.0/sum(manager_level[temp['manager_id']])
            b[j]=manager_level[temp['manager_id']][1]*1.0/sum(manager_level[temp['manager_id']])
            c[j]=manager_level[temp['manager_id']][2]*1.0/sum(manager_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c
a_mean = np.mean(a)
b_mean = np.mean(b)
c_mean = np.mean(c)

a=[]
b=[]
c=[]
manager_level={}
for j in train_df['manager_id'].values:
    manager_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        manager_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        manager_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        manager_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in manager_level.keys():
        a.append(a_mean)
        b.append(b_mean)
        c.append(c_mean)
    else:
        man_level_sum = sum(manager_level[i])
        a.append(manager_level[i][0]*1.0/man_level_sum)
        b.append(manager_level[i][1]*1.0/man_level_sum)
        c.append(manager_level[i][2]*1.0/man_level_sum)

test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')
           
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
