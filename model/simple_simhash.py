
# coding: utf-8

import pandas as pd
import numpy as np
from simhash import Simhash,SimhashIndex
import jieba

sub = pd.read_csv('../input/smp_sample.csv')

with open('../input/源句子.txt','rb') as train_file:
    train = pd.read_table(train_file,names=['id','content'],header=None,delimiter='\n')
    
with open('../input/待判定句子.txt','rb') as test_file:
    test = pd.read_table(test_file,names=['id','content'],header=None,delimiter='\n')
print(train.shape,test.shape)

train['source']= "train"
test['source'] = "test"
data = pd.concat([train,test])

data['content'] = data['id'].apply(lambda x : x[9:])
data['id'] = data['id'].apply(lambda x : x[:8])
data['content'] = data['content'].map(str)

stopwords_path = '../input/stop_words.txt' 



def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr="/ ".join(seg_list)
    f_stop = open(stopwords_path)
    try:
        f_stop_text = f_stop.read( )
        #f_stop_text=unicode(f_stop_text,'utf-8')
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return ''.join(mywordlist)

#data.head()['content'].apply(lambda x:jiebaclearText(str(x)))

data['content'] = data['content'].apply(lambda x:jiebaclearText(str(x)))
data['simhash'] = data['content'].apply(lambda x:Simhash(x).value)

train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source',],axis=1,inplace=True)

objs = [(row["id"], Simhash(row["content"])) for index, row in train.iterrows()]

index = SimhashIndex(objs, k=12)
test['result'] = test['content'].apply(lambda x:index.get_near_dups(Simhash(x)))

sub['result']=test['result']
sub.to_csv('../output/simhash.csv',index=False)
