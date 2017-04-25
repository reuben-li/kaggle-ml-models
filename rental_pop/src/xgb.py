"""
XGB for rental-listing kaggle competition
"""

from __future__ import print_function
import sys
from scipy import sparse
from sklearn import tree
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from Levenshtein import distance
import numpy as np
import pandas as pd
import math

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(0)

TOLERANCE = 50
EPOCHS = 1300
INTEREST = ['low', 'medium', 'high']

def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho

def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi

def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"

    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def operate_on_coordinates(tr_df, te_df):
    for df in [tr_df, te_df]:

        df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)

        #rotations
        print("calculating rotations")
        for angle in [15,30,45,60]:
            df = add_rotation(angle, df)

    return tr_df, te_df


def correct(df, train=True, verbose=False):
    """ Adjust results to data distribution """

    tau_train = {
        'low': 0.694683,
        'medium': 0.227529,
        'high': 0.077788,
    }

    tau_test = {
        'low': 0.69195995,
        'medium': 0.23108864,
        'high': 0.07695141,
    }

    if train:
        tau = tau_train
    else:
        tau = tau_test

    index = df['listing_id']
    df_sum = df[INTEREST].sum(axis=1)
    df_correct = df[INTEREST].copy()

    if verbose:
        dupli = df_correct.mean()
        a = [tau[k] / dupli[k]  for k in INTEREST]
        print(a)

    for c in INTEREST:
        df_correct[c] /= df_sum

    for i in range(20):
        for c in INTEREST:
            df_correct[c] *= tau[c] / df_correct[c].mean()

        df_sum = df_correct.sum(axis=1)

        for c in INTEREST:
            df_correct[c] /= df_sum

    if verbose:
        dupli = df_correct.mean()
        a = [tau[k] / dupli[k]  for k in INTEREST]
        print(a)

    df_correct = pd.concat([index, df_correct], axis=1)
    return df_correct

def loaddata():
    """ load data """
    train_df = pd.read_json('../input/train.json', convert_dates=['created'])
    test_df = pd.read_json('../input/test.json', convert_dates=['created'])
    image_date = pd.read_csv('../input/listing_image_time.csv')

    return train_df, test_df, image_date

def prep_features(train_df, test_df):
    """ prepare features """

    def binner(feat, bins, dyn=True):
        """ create bins for continuous features """
        out, featbins = pd.qcut(train_df[feat], bins, retbins=True, labels=False)
        train_df[feat + '_bin'] = out.astype(float)
        if dyn:
            test_df[feat + '_bin'] = pd.cut(test_df[feat], featbins, labels=False).astype(float)
        else:
            test_df[feat + '_bin'] = pd.qcut(test_df[feat], bins, labels=False).astype(float)

    def med(feat):
        """ get median values from feature """
        comb = pd.concat([train_df, test_df])
        feat_uniq = comb[feat].unique()
        feat_med = {}
        for i in feat_uniq:
            a = comb[comb[feat] == i].price.median()
            feat_med[i] = a
        train_df[feat + '_med'] = train_df[feat].apply(lambda x: feat_med[x])
        train_df[feat + '_value'] = train_df[feat + '_med'] - train_df['price']
        test_df[feat + '_med'] = test_df[feat].apply(lambda x: feat_med[x])
        test_df[feat + '_value'] = test_df[feat + '_med'] - test_df['price']

    med('bedrooms')
    medlat = train_df['latitude'].median()
    medlon = train_df['longitude'].median()

    #reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    #def try_and_find_nr(description):
    #    if reg.match(description) is None:
    #        return 0
    #    return 1

    for data in [train_df, test_df]:
        # basic features
        data['price_t'] = data['price'] / data['bedrooms']
        data['room_sum'] = data['bedrooms'] + data['bathrooms']
        data['layout'] = data['bathrooms'] + train_df['bedrooms']
        data['distance'] = (abs(data['longitude'] - medlon)**2 + abs(data['latitude'] - medlat)**2)**0.5
        data['chars'] = len(data['description'])
        data['exclaim'] = [x.count('!') for x in data['description']]
        data['shock'] = data['exclaim'] / data['chars'] * 100
        data['halfbr'] = data['bathrooms'].apply(lambda x: 0 if round(x) == x else 1)
        data['toobig'] = data['bedrooms'].apply(lambda x: 1 if x > 4 else 0)
        data['month'] = data['created'].apply(lambda x: x.month).astype(object)
        data['yearmonth'] = data['created'].apply(lambda x: str(x.year) + '_' + str(x.month))
        data['day'] = data['created'].apply(lambda x: x.dayofweek)
        data['weekend'] = data['day'].apply(lambda x: 1 if x == 5 or x == 6 else 0)
        data['friday'] = data['day'].apply(lambda x: 1 if x == 4 else 0)
        data['num_photos'] = data['photos'].apply(len)
        data['nophoto'] = data['num_photos'].apply(lambda x: 1 if x == 0 else 0)
        data['num_features'] = data['features'].apply(len)
        data['num_description_words'] = data['description'].apply(lambda x: len(x.split(' ')))
        data['upper_case'] = data['description'].apply(lambda x: sum(1 for i in x if i.isupper()))
        data['upper_percent'] = data['upper_case']*100.0/data['description'].apply(lambda x: len(x))
        data['address_distance'] = data[['street_address', 'display_address']].apply(lambda x: distance(*x), axis=1)
        #data['num_redacted'] = 0
        #data['num_redacted'].ix[data['description'].str.contains('website_redacted')] = 1
        data['num_lines'] = data['description'].apply(lambda x: x.count('<br /><br />'))
        data['num_email'] = 0
        data['num_email'].ix[data['description'].str.contains('@')] = 1
        #data['num_phone'] = data['description'].apply(try_and_find_nr)

    binner('longitude', 20)
    binner('price_t', 7, False)

    # cross variables
    abc_list = []
    classes=12
    for i in xrange(97, 123):
        abc_list.append(str(chr(i)))
    train_lon, lon_bins = pd.qcut(train_df['longitude'], classes, retbins=True, labels=abc_list[0:classes])
    train_lat, lat_bins = pd.qcut(train_df['latitude'], classes, retbins=True, labels=abc_list[0:classes])
    train_lon = train_lon.astype(object)
    train_lat = train_lat.astype(object)
    train_df['grid'] = train_lon + train_lat
    test_lon = pd.cut(test_df['longitude'], lon_bins, labels=abc_list[0:classes]).astype(object)
    test_lat = pd.cut(test_df['latitude'], lat_bins, labels=abc_list[0:classes]).astype(object)
    test_df['grid'] = test_lon + test_lat

    le = preprocessing.LabelEncoder()
    le.fit(train_df['grid'].append(test_df['grid']))
    train_df['grid'] = le.transform(train_df['grid'])
    test_df['grid'] = le.transform(test_df['grid'])


    # rename columns so you can join tables later on
    image_date.columns = ['listing_id', 'time_stamp']

    # reassign the only one timestamp from April, all others from Oct/Nov
    image_date.loc[80240,'time_stamp'] = 1478129766
    image_date['img_date'] = pd.to_datetime(image_date['time_stamp'], unit='s')
    image_date['img_days_passed'] = (image_date['img_date'].max() - image_date['img_date']).astype('timedelta64[D]').astype(int)
    image_date['img_date_month'] = image_date['img_date'].dt.month
    image_date['img_date_week'] = image_date['img_date'].dt.week
    image_date['img_date_day'] = image_date['img_date'].dt.day
    image_date['img_date_dayofweek'] = image_date['img_date'].dt.dayofweek
    image_date['img_date_dayofyear'] = image_date['img_date'].dt.dayofyear
    image_date['img_date_hour'] = image_date['img_date'].dt.hour
    image_date['img_date_monthBeginMidEnd'] = image_date['img_date_day'].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

    train_df = pd.merge(train_df, image_date, on='listing_id', how='left')
    test_df = pd.merge(test_df, image_date, on='listing_id', how='left')

    binner('time_stamp', 20)

    print('predicting price profile')
    clf = tree.DecisionTreeClassifier()
    params = ['bedrooms', 'bathrooms', 'num_features', 'grid']
    clf = clf.fit(train_df[params], train_df['price'])
    train_df['exp_price'] = pd.DataFrame(clf.predict(train_df[params]).tolist()).set_index(train_df.index)
    train_df['overprice'] = train_df['price'] - train_df['exp_price']

    test_df['exp_price'] = pd.DataFrame(clf.predict(test_df[params]).tolist()).set_index(test_df.index)
    test_df['overprice'] = test_df['price'] - test_df['exp_price']

    return train_df, test_df

print('Loading data')
train_df, test_df, image_date = loaddata()

print('Extracting features')
train_df, test_df = prep_features(train_df, test_df)
#train_df, test_df = operate_on_coordinates(train_df, test_df)

features_to_use = ['latitude', 'longitude_bin', 'bathrooms', 'bedrooms', 'address_distance',
                   'price', 'price_t', 'num_photos', 'num_features', 'num_description_words',
                   'listing_id', 'time_stamp', 'img_days_passed', 'img_date_month', 'img_date_week',
                   'img_date_day', 'img_date_dayofweek', 'img_date_hour', 'nophoto',
                   'img_date_monthBeginMidEnd', 'upper_case', 'upper_percent', 'halfbr',
                   'exp_price', 'price_t_bin', 'layout', 'distance', 'bedrooms_value',
                   'num_lines', 'num_email'
                   #'num_rho', 'num_phi', 'num_rot15_X', 'num_rot30_X', 'num_rot45_X', 'num_rot60_X',
                   ]

categorical = ['display_address', 'manager_id', 'building_id', 'street_address']

print('Transforming categorical data')

for f in categorical:
    print(f)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values) + list(test_df[f].values))
    train_df[f] = lbl.transform(list(train_df[f].values))
    test_df[f] = lbl.transform(list(test_df[f].values))
    features_to_use.append(f)

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=EPOCHS):
    """ extreme gradient boost """
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
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

print('Manager ID cv statistics')

def cvstats():
    for group in ['manager_id']:
        if not np.DataSource('train_abc.npy'):
            print('train_abc.npy not found')
            index = list(range(train_df.shape[0]))
            random.shuffle(index)
            a = [np.nan]*len(train_df)
            b = [np.nan]*len(train_df)
            c = [np.nan]*len(train_df)

            for i in range(5):
                manager_level={}
                for j in train_df[group].values:
                    manager_level[j] = [0, 0, 0]
                test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
                train_index=list(set(index).difference(test_index))
                for j in train_index:
                    temp = train_df.iloc[j]
                    if temp['interest_level'] == 'low':
                        manager_level[temp[group]][0] += 1
                    if temp['interest_level'] == 'medium':
                        manager_level[temp[group]][1] += 1
                    if temp['interest_level'] == 'high':
                        manager_level[temp[group]][2] += 1
                for j in test_index:
                    temp = train_df.iloc[j]
                    if sum(manager_level[temp[group]]) != 0:
                        a[j]=manager_level[temp[group]][0] * 1.0 / sum(manager_level[temp[group]])
                        b[j]=manager_level[temp[group]][1] * 1.0 / sum(manager_level[temp[group]])
                        c[j]=manager_level[temp[group]][2] * 1.0 / sum(manager_level[temp[group]])
            np.save('train_abc', (a, b, c))
        else:
            print('loading train_abc.npy')
            a, b, c = np.load('train_abc.npy')

        train_df[group + '_low'] = a
        train_df[group + '_medium'] = b
        train_df[group + '_high'] = c
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        c_mean = np.mean(c)

        if not np.DataSource('test_abc.npy'):
            print('test_abc.npy not found')
            a = []
            b = []
            c = []
            manager_level = {}
            for j in train_df[group].values:
                manager_level[j]=[0, 0, 0]
            for j in range(train_df.shape[0]):
                temp=train_df.iloc[j]
                if temp['interest_level'] == 'low':
                    manager_level[temp[group]][0]+=1
                if temp['interest_level'] == 'medium':
                    manager_level[temp[group]][1]+=1
                if temp['interest_level'] == 'high':
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
            np.save('test_abc', (a, b, c))
        else:
            print('loading test_abc.npy')
            a, b, c = np.load('test_abc.npy')

        test_df[group + '_low'] = a
        test_df[group + '_medium'] = b
        test_df[group + '_high'] = c

        features_to_use.append(group + '_low')
        features_to_use.append(group + '_medium')
        features_to_use.append(group + '_high')

cvstats()
train_df['features'] = train_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
test_df['features'] = test_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=180)
tr_sparse = tfidf.fit_transform(train_df['features'])
te_sparse = tfidf.transform(test_df['features'])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print('Training')

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
out_df.columns = ['high', 'medium', 'low']
out_df['listing_id'] = test_df.listing_id.values
out_df = correct(out_df, train=False, verbose=True)
out_df.to_csv('../output/xgb_output.csv', index=False)
