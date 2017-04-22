import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
from sklearn import tree
from Levenshtein import distance
import random

import sys
reload(sys)
sys.setdefaultencoding('utf8')

DATA_PATH='../input/'
TOLERANCE=30

def load_data_sparse(data_path=DATA_PATH):

    train_file = data_path + "train.json"
    test_file = data_path + "test.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    image_date = pd.read_csv(data_path + "listing_image_time.csv")
    image_date.columns = ["listing_id", "time_stamp"]

    # original features
    features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude",
                       "price"]

    # feature engineering on both test and train datatest
    for df in [train_df, test_df]:

        # count of photos
        df["num_photos"] = df["photos"].apply(len)
        df["nophoto"] = df["num_photos"].apply(lambda x: 1 if x == 0 else 0)

        # count of house features
        df["num_features"] = df["features"].apply(len)

        # count of words present in description column
        df["num_words"] = df["description"].apply(lambda x: len(x.split(' ')))
        df["uppercase"] = df["description"].apply(lambda x: sum(1 for i in x if i.isupper()))
        df["upper_percent"] = df["uppercase"] * 100.0 / df["num_words"]

        # room related
        df["rooms"] = df["bathrooms"] + df["bedrooms"]
        df["toobig"] = df["bedrooms"].apply(lambda x: 1 if x > 4 else 0)
        df["halfbr"] = df["bathrooms"].apply(lambda x: 0 if round(x) == x else 1)

        # price related
        df["price_t"] = (df["price"]) / (df["rooms"] + 1.0)
        df["price_lat"] = (df["price"]) / (df["latitude"] + 1.0)
        df["price_lon"] = (df["price"]) / (df["longitude"] - 1.0)
        df["rooms"] = df["rooms"].apply(lambda x: str(x) if float(x) < 9.5 else '10')

        # difference between addresses
        df["add_dist"] = df[["street_address", "display_address"]].apply(lambda x: distance(*x), axis=1)

        # date related
        df["created"] = pd.to_datetime(df["created"])
        df["c_month"] = df["created"].dt.month
        df["c_day"] = df["created"].dt.day
        df["c_hour"] = df["created"].dt.hour
        df["total_days"] = (df["c_month"] -4.0) * 30 + df["c_day"] + df["c_hour"] / 25.0
        df["diff_rank"] = df["total_days"] / (df["listing_id"] - 68119576.0)
        df["weekend"] = df["c_day"].apply(lambda x: 1 if x in [5, 6] else 0)

    # magic variable!
    train_df = pd.merge(train_df, image_date, on="listing_id", how="left")
    test_df = pd.merge(test_df, image_date, on="listing_id", how="left")
    # normalize listing_id
    train_df["listing_id"] = train_df["listing_id"] - 68119576.0
    test_df["listing_id"] = test_df["listing_id"] - 68119576.0

    # grid encoding
    abc_list = []
    num = 12

    for i in xrange(97, 123):
        abc_list.append(str(chr(i)))
    train_lon, lon_bins = pd.qcut(train_df["longitude"], num, retbins=True,
                              labels=abc_list[0:num])
    train_lat, lat_bins = pd.qcut(train_df["latitude"], num, retbins=True,
                              labels=abc_list[0:num])
    train_lon = train_lon.astype(object)
    train_lat = train_lat.astype(object)
    train_df["grid"] = train_lon + train_lat

    test_lon = pd.cut(test_df["longitude"], lon_bins, labels=abc_list[0:num]).astype(object)
    test_lat = pd.cut(test_df["latitude"], lat_bins, labels=abc_list[0:num]).astype(object)
    test_df["grid"] = test_lon + test_lat

    le = LabelEncoder()
    le.fit(train_df["grid"].append(test_df["grid"]))
    train_df["grid"] = le.transform(train_df["grid"])
    test_df["grid"] = le.transform(test_df["grid"])

    # price level estimation
    print('predicting price profile')

    clf = tree.DecisionTreeClassifier()
    xparams = ['bedrooms', 'bathrooms', 'num_features', 'grid']
    clf = clf.fit(train_df[xparams], train_df['price'])
    train_df["exp_price"] = pd.DataFrame(clf.predict(train_df[xparams])
                                .tolist()).set_index(train_df.index)
    train_df["overprice"] = train_df["price"] - train_df["exp_price"]

    test_df["exp_price"] = pd.DataFrame(clf.predict(test_df[xparams])
                               .tolist()).set_index(test_df.index)
    test_df["overprice"] = test_df["price"] - test_df["exp_price"]

    # price group
    train_pg, pricet_bins = pd.qcut(train_df["price_t"], num,
                                                retbins=True, labels=abc_list[0:num])
    test_pg = pd.cut(df["price_t"], pricet_bins, labels=abc_list[0:num])
    train_df['pricet_group'] = train_pg.astype(object)
    test_df['pricet_group'] = test_pg.astype(object)

    categorical = ["grid", "display_address", "manager_id", "building_id",
                   "street_address", "rooms", "pricet_group"]

    # generate categorical cross features
    print("generating categorical cross features")

    lencat = len(categorical)

    for f in range (0, lencat):
        for s in range (f+1,lencat):
            for df in [train_df, test_df]:
                print(categorical[s])
                df[categorical[f]] = str(df[categorical[f]])
                df[categorical[f] + "_" + categorical[s]] = df[categorical[f]] + "_" + df[categorical[s]]
            categorical.append(categorical[f] + "_" +categorical[s])

    # add continuous features
    features_to_use.extend(["overprice", "num_photos", "num_features",
                            "num_words", "c_month", "c_day", "listing_id",
                            "c_hour", "total_days", "diff_rank", "uppercase",
                            "upper_percent", "rooms", "add_dist", "time_stamp",
                            "price_t","price_lat","price_lon", "toobig",
                            "halfbr", "weekend"])

    result = pd.concat([train_df,test_df])

    for f in categorical:
        #if train_df[f].dtype=='object':

            cases=defaultdict(int)
            temp=np.array(result[f]).tolist()
            for k in temp:
                cases[k]+=1
            print f, len(cases)

            train_df[f]=train_df[f].apply(lambda x: cases[x])
            test_df[f]=test_df[f].apply(lambda x: cases[x])

            features_to_use.append(f)

    # manager_id stats
    print('manager_id stats')    

    index=list(range(train_df.shape[0]))
    random.seed(0)
    random.shuffle(index)
    a=[np.nan]*len(train_df)
    b=[np.nan]*len(train_df)
    c=[np.nan]*len(train_df)

    group = 'manager_id'
    for i in range(5):
        manager_level={}
        for j in train_df[group].values:
            manager_level[j]=[0,0,0]
        test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df.iloc[j]
            if temp['interest_level'] == 'low':
                manager_level[temp[group]][0] += 1
            if temp['interest_level'] == 'medium':
                manager_level[temp[group]][1] += 1
            if temp['interest_level'] == 'high':
                manager_level[temp[group]][2] += 1
        for j in test_index:
            temp=train_df.iloc[j]
            if sum(manager_level[temp[group]]) != 0:
                a[j]=manager_level[temp[group]][0] * 1.0 / sum(manager_level[temp[group]])
                b[j]=manager_level[temp[group]][1] * 1.0 / sum(manager_level[temp[group]])
                c[j]=manager_level[temp[group]][2] * 1.0 / sum(manager_level[temp[group]])
    train_df['manager_level_low'] = a
    train_df['manager_level_medium'] = b
    train_df['manager_level_high'] = c
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c_mean = np.mean(c)
    
    a = []
    b = []
    c = []
    manager_level = {}
    for j in train_df[group].values:
        manager_level[j]=[0,0,0]
    for j in range(train_df.shape[0]):
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            manager_level[temp[group]][0]+=1
        if temp['interest_level']=='medium':
            manager_level[temp[group]][1]+=1
        if temp['interest_level']=='high':
            manager_level[temp[group]][2]+=1
    
    for i in test_df[group].values:
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
    features_to_use.append('manager_level_high')
    features_to_use.append('manager_level_medium')


    train_df['features'] =  train_df['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_df['features'] =test_df['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

    train_df['description'] =  train_df['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc")
    test_df['description'] =test_df['description'].apply(lambda x: str(x).encode('utf-8') if len(x)>2 else "nulldesc")

    tfidfdesc=TfidfVectorizer(min_df=20, max_features=50, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), use_idf=False,smooth_idf=False,
    sublinear_tf=True, stop_words = 'english')

    tfidf = CountVectorizer(stop_words='english', max_features=200)

    te_sparse = tfidf.fit_transform (test_df["features"])
    tr_sparse = tfidf.transform(train_df["features"])

    te_sparsed = tfidfdesc. fit_transform (test_df["description"])
    tr_sparsed = tfidfdesc.transform(train_df["description"])
    print(features_to_use)


    train_X = sparse.hstack([train_df[features_to_use], tr_sparse,tr_sparsed]).tocsr()#
    test_X = sparse.hstack([test_df[features_to_use], te_sparse,te_sparsed]).tocsr()#

    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    ids= test_df.listing_id.values
    print(train_X.shape, test_X.shape)
    return train_X,test_X,train_y,ids

#create average value of the target variabe given a categorical feature
def convert_dataset_to_avg(xc,yc,xt, rounding=2,cols=None):
    xc=xc.tolist()
    xt=xt.tolist()
    yc=yc.tolist()
    if cols==None:
        cols=[k for k in range(0,len(xc[0]))]
    woe=[ [0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good=[]
    bads=[]
    for col in cols:
        dictsgoouds=defaultdict(int)
        dictsbads=defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)
    total_count=0.0
    total_sum =0.0

    for a in range (0,len(xc)):
        target=yc[a]
        total_sum+=target
        total_count+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            good[j][round(xc[a][col],rounding)]+=target
            bads[j][round(xc[a][col],rounding)]+=1.0
    #print(total_goods,total_bads)

    for a in range (0,len(xt)):
        for j in range(0,len(cols)):
            col=cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j]=float(good[j][round(xt[a][col],rounding)])/float(bads[j][round(xt[a][col],rounding)])
            else :
                 woe[a][j]=round(total_sum/total_count)
    return woe


#converts the select categorical features to numerical via creating averages based on the target variable within kfold.

def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):

    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]
    #print("it is not!!")
    X=X.tolist()
    Xt=Xt.tolist()
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]

    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns)

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):
            woetest[real_index][j]=Xt[real_index][j]

    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]

    return np.array(woetrain), np.array(woetest)


def main():

        #training and test files, created using SRK's python script
        train_file="train_stacknet.csv"
        test_file="test_stacknet.csv"

        X, X_test, y, ids = load_data_sparse(data_path=DATA_PATH)
        ids= np.array([int(k)+68119576 for k in ids ])
        print(X.shape, X_test.shape)

        #create to numpy arrays (dense format)
        X=X.toarray()
        X_test=X_test.toarray()

        print ("scaling")
        #scale the data
        stda=StandardScaler()
        print(np.any(np.isnan(X_test)))
        X_test=stda.fit_transform(X_test)
        X=stda.transform(X)

        CO=[0,14,21] # columns to create averages on

        #Create Arrays for meta
        train_stacker=[ [0.0 for s in range(3)]  for k in range (0,(X.shape[0])) ]
        test_stacker=[[0.0 for s in range(3)]   for k in range (0,(X_test.shape[0]))]

        number_of_folds=5 # number of folds to use
        print("kfolder")
        #cerate 5 fold object
        mean_logloss = 0.0
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=15)

        #xgboost_params
        param = {}
        param['booster']='gbtree'
        param['objective'] = 'multi:softprob'
        param['bst:eta'] = 0.03
        param['seed']=  1
        param['bst:max_depth'] = 6
        param['bst:min_child_weight']= 1.
        param['silent'] =  1
        param['nthread'] = 12 # put more if you have
        param['bst:subsample'] = 0.7
        param['gamma'] = 1.0
        param['colsample_bytree']= 1.0
        param['num_parallel_tree']= 3
        param['colsample_bylevel']= 0.7
        param['lambda']=5
        param['num_class']= 3
        param['eval_metric'] = "mlogloss"
        # num_rounds = num_rounds

        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                #create past averages for some fetaures
                W_train,W_cv=convert_to_avg(X_train,y_train, X_cv, seed=1, cvals=5, roundings=2, columns=CO)
                W_train=np.column_stack((X_train,W_train[:,CO]))
                W_cv=np.column_stack((X_cv,W_cv[:,CO]))
                print (" train size: %d. test size: %d, cols: %d " % ((W_train.shape[0]) ,(W_cv.shape[0]) ,(W_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(csr_matrix(W_train), label=np.array(y_train),missing =-999.0)
                X1cv=xgb.DMatrix(csr_matrix(W_cv), label=np.array(y_cv), missing =-999.0)
                watchlist = [ (X1,'train'), (X1cv, 'test') ]
                bst = xgb.train(param.items(), X1, 1000, watchlist, early_stopping_rounds=TOLERANCE)
                #predictions
                predictions = bst.predict(X1cv)
                preds=predictions.reshape( W_cv.shape[0], 3)

                #scalepreds(preds)
                logs = log_loss(y_cv,preds)
                print "size train: %d size cv: %d loglikelihood (fold %d/%d): %f" % ((W_train.shape[0]), (W_cv.shape[0]), i + 1, number_of_folds, logs)

                mean_logloss += logs
                #save the results
                no=0
                for real_index in test_index:
                    for d in range (0,3):
                        train_stacker[real_index][d]=(preds[no][d])
                    no+=1
                i+=1
        mean_logloss/=number_of_folds
        print (" Average Lolikelihood: %f" % (mean_logloss) )

        #calculating averages for the train data
        W,W_test=convert_to_avg(X,y, X_test, seed=1, cvals=5, roundings=2, columns=CO)
        W=np.column_stack((X,W[:,CO]))
        W_test=np.column_stack((X_test,W_test[:,CO]))
        #X_test=np.column_stack((X_test,woe_cv))
        print (" making test predictions ")

        X1=xgb.DMatrix(csr_matrix(W), label=np.array(y) , missing =-999.0)
        X1cv=xgb.DMatrix(csr_matrix(W_test), missing =-999.0)
        bst = xgb.train(param.items(), X1, 1000)
        predictions = bst.predict(X1cv)
        preds=predictions.reshape( W_test.shape[0], 3)

        for pr in range (0,len(preds)):
                for d in range (0,3):
                    test_stacker[pr][d]=(preds[pr][d])

        print ("merging columns")
        #stack xgboost predictions
        X=np.column_stack((X,train_stacker))
        # stack id to test
        X_test=np.column_stack((X_test,test_stacker))

        # stack target to train
        X=np.column_stack((y,X))
        # stack id to test
        X_test=np.column_stack((ids,X_test))

        #export to txt files (, del.)
        print ("exporting files")
        np.savetxt(train_file, X, delimiter=",", fmt='%.5f')
        np.savetxt(test_file, X_test, delimiter=",", fmt='%.5f')

        print("Write results...")
        output_file = "submission_"+str( (mean_logloss ))+".csv"
        print("Writing submission to %s" % output_file)
        f = open(output_file, "w")
        f.write("listing_id,high,medium,low\n")# the header
        for g in range(0, len(test_stacker))  :
          f.write("%s" % (ids[g]))
          for prediction in test_stacker[g]:
             f.write(",%f" % (prediction))
          f.write("\n")
        f.close()
        print("Done.")

if __name__=="__main__":
  main()
