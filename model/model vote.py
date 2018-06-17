import pandas as pd
import numpy as np

sen2vec = pd.read_csv('../output/sen2vec_100replace.csv')
sen2vec.rename(columns = {'result':'sen2vec'},inplace=True)

freq = pd.read_csv('../output/frequency_threshold50.csv')
freq.rename(columns = {'result':'freq'},inplace=True)

print(sen2vec.head())
print(freq.head())

comb = pd.merge(sen2vec,freq)
print(comb.head())

def compare(x,y):
    if x==y:
        return x
    else:
        return

result = comb.apply(lambda row:compare(row['sen2vec'],row['freq']),axis=1)
comb['result'] = result
print(comb['result'].count())
comb.drop(['sen2vec','freq'],axis=1).to_csv('../output/comb_freq_threshold50.csv',index=False)