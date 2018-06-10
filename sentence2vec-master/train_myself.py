import pandas as pd
import re
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

data = pd.read_csv('../feature/data_cutted.csv')

# get array of titles
titles = data['content'].values.tolist()

# tokenize the each title
tok_titles = [word_tokenize(str(title)) for title in titles]

# refer to here for all parameters:
# https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(tok_titles, sg=1, size=200, window=5, min_count=5, workers=4,
                 iter=100)

# save model to file
model.save('../feature/content_200.model')
