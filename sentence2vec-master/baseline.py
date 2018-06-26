# coding: utf-8

from lib.sentence2vec import Sentence2Vec
import pandas as pd


data = pd.read_csv('../feature/data_cutted.csv')
model = Sentence2Vec('../feature/content_small.model')

sen2vec = pd.DataFrame(data['content'].apply(lambda x:model.get_vector(str(x))).tolist())

data = pd.concat([data,sen2vec],axis=1)
data.drop(['content'],axis=1,inplace=True)

data.to_csv('../feature/data_vec.csv',index=False)
data = pd.read_csv('../feature/data_vec.csv')

train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop(['source'],axis=1,inplace=True)
test.drop(['source'],axis=1,inplace=True)


X = train.drop(['id'],axis=1)
y = train['id']

print('load model.')
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y) 

feature = test.drop(['id'],axis=1)
result = neigh.predict(feature)

sub = pd.read_csv('../input/smp_sample.csv')

sub = pd.DataFrame({'0':test['id'],'1':result})
sub.columns=['test_id','result']
#sub.replace({'﻿mhY5opF':'?mhY5opF4'},inplace=True) 
print(sub.head())

sub.to_csv('../output/sen2vec_300small_knn.csv',index=False)