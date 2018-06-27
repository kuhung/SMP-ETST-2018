import jieba
import pandas as pd


#train_file=open('../input/源句子.txt','rb')
with open('../input/源句子.txt','rb') as train_file:
    train = pd.read_table(train_file,names=['id','content'],header=None,delimiter='\n')
#test_file=open('../input/待判定句子.txt','rb')
with open('../input/待判定句子.txt','rb') as test_file:
    test = pd.read_table(test_file,names=['id','content'],header=None,delimiter='\n')

print(train.shape,test.shape)

train['source']= "train"
test['source'] = "test"
data = pd.concat([train,test])

data['content'] = data['id'].apply(lambda x : x[9:])
data['id'] = data['id'].apply(lambda x : x[:8])
data['content'] = data['content'].map(str)
data.replace({'﻿mhY5opF':'?mhY5opF4'},inplace=True) 
data['length'] = data['content'].apply(lambda x:len(str(x)))

stopwords_path = '../input/stop_words.txt' 

def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text.lower(), cut_all=False)
    liststr="/ ".join(seg_list)
    f_stop = open(stopwords_path)
    try:
        f_stop_text = f_stop.read( )
        #f_stop_text=str(f_stop_text,'utf-8')
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return ''.join(mywordlist)

data['content'] = data['content'].apply(lambda x:jiebaclearText(str(x)))

data.to_csv('../feature/data_cutted.csv',index=False)