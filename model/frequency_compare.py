import pandas as pd
from collections import Counter
import jieba
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('../feature/data_cutted.csv')

le = preprocessing.LabelEncoder()
data['id'] = le.fit_transform(data['id'])

def get_words(txt):
    txt = jieba.cut(txt)  
    c = Counter()  
    for x in txt:  
        if len(x)>1 and x != '\r\n':  
            c[x] += 1
    return c

data['vector_content'] = data['content'].apply(lambda x: get_words(str(x)))

def get_cosine_new(vec1, vec2):
    try:
        present_count = 0
        total_count = 0
        for word,count in vec1.items():
            total_count += count
            if word in vec2:
                present_count += count
        return present_count/max(total_count,1)
    except:
        return 0.0

def find_relation(inp_df):
    relation_df = []
    train = inp_df.loc[data['source']=='train'].reset_index()
    test = inp_df.loc[data['source']=='test'].reset_index()
    
    for i in range(len(test)):
        for j in range(len(train)):
            rel_df = dict()
            try:
                rel_df["cosine_abstract"] = get_cosine_new(test['vector_content'][i],train['vector_content'][j])
            except:
                rel_df["cosine_abstract"] = 0.0
            
            if rel_df["cosine_abstract"] > 0.0:
                rel_df["id"] = test["id"][i]
                rel_df["related_id"] = train["id"][j]
                relation_df.append(rel_df)
            else:
                pass
    final_df = pd.DataFrame(relation_df)
    return final_df


frequency_vec = find_relation(data)


frequency_vec['id'] = le.inverse_transform(frequency_vec['id'])
frequency_vec['related_id'] = le.inverse_transform(frequency_vec['related_id'])

frequency_vec.to_csv('../feature/cosine_abstract.csv',index=False)
frequency_vec = pd.read_csv('../feature/cosine_abstract.csv')

print(frequency_vec.shape)
print(frequency_vec.describe())

frequency_pivot = frequency_vec.set_index(['id','related_id']).unstack()
#frequency_pivot.reset_index().to_csv('../feature/frequency_pivot.csv',index=False)
#frequency_pivot = pd.read_csv('../feature/frequency_pivot.csv',skiprows=1 )
result = frequency_pivot.idxmax(axis = 1).apply(lambda x : str(x)[21:29])
result_pro = np.max(frequency_pivot,axis=1)

sub = pd.DataFrame({'test_id':frequency_pivot.index,'result':result,'pro':result_pro})
sub.replace({'?mhY5opF':'?mhY5opF4'},inplace=True) 

#sub.describe()
#TODO: threshold auto set

def threshold(result,distance):
    if distance > 0.545455:
        return result
    else:
        return 


sub['result'] = sub.apply(lambda row: threshold(row['result'],row['pro']),axis=1)
sub.drop('pro',axis=1).to_csv('../output/frequency_threshold50.csv',index=False)
