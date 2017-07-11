import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import os

print('Loading data ...')

train_p = '../input/train_p'
prop_p = '../input/prop_p'
sample_p= '../input/sample_p'

if os.path.exists(train_p):
   train = pd.read_pickle(train_p)
else:
   train = pd.read_csv('../input/train_2016_v2.csv')
   train.to_pickle(train_p)

if os.path.exists(prop_p):
    prop = pd.read_pickle(prop_p)
else:
    prop = pd.read_csv('../input/properties_2016.csv')
    prop.to_pickle(prop_p)

if os.path.exists(sample_p):
    sample = pd.read_pickle(sample_p)
else:
    sample = pd.read_csv('../input/sample_submission.csv')
    sample.to_pickle(sample_p)

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Feature engineering ...')

prop['bedbathratio'] = prop['bedroomcnt'] / prop['bathroomcnt']

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('results/xgb_starter.csv', index=False, float_format='%.4f') # Thanks to @inversion
