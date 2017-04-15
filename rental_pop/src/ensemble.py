import pandas as pd
import numpy as np
from scipy import stats

PATH = '../output/'
MODELS = ['address_distance.csv', 'month.csv', 'puregrid.csv', 'original.csv']


main = pd.read_csv(PATH + MODELS.pop())

for m in MODELS:
  low = pd.read_csv(PATH + m).low
  med = pd.read_csv(PATH + m).medium
  high = pd.read_csv(PATH + m).high
  main.low = main['low'].add(low)
  main.med = main['medium'].add(med)
  main.high = main['high'].add(high) 

main.to_csv('../output/ensemble.csv', index=False)
