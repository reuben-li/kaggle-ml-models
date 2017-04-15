import pandas as pd
import numpy as np
from scipy import stats

PATH = '../output/'
MODELS = ['address_distance', 'month', 'puregrid', 'original']

model_list = {}

for i in xrange(len(MODELS)):
    model_list[MODELS[i]] = pd.read_csv(PATH + MODELS[i] + '.csv')

allm = pd.concat(model_list.values())
ensem = allm.groupby(level=0).mean()

ensem.to_csv('../output/ensemble.csv', index=False)
