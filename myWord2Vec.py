
#đọc dữ liệu
import pandas as pd

pathFile = "Word2VecData.inp"
df = pd.read_csv(pathFile, names=["row"]).dropna()
print(df.head(5))

#Làm sạch dữ liệu
import re
def transform_row(row):
	#Xóa các ký tự không phải chữ và số
	row = re.sub(r'\W+', " ", row)
	#Xóa các ký tự là số
	row = re.sub(r"[0-9]+", " ", row)
	#Chuẩn hóa số dấu cách
	row = re.sub(r" {2,}", " ", row)
	#Xóa khoảng trắng ở đầu và cuối
	row = row.strip()

	return row 

df["row"] = df.row.apply(transform_row)
print(df.head(5))

#tách từ bằng ngrams
from nltk import ngrams

def split_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]

#tách thành các tiếng
df["1gram"] = df.row.apply(lambda t: split_ngram(t, 1))
print(df.head(5))

#tách thành các từ
df["2gram"] = df.row.apply(lambda t: split_ngram(t, 2))
print(df.head(5))

df["context"] = df["1gram"] + df["2gram"]
train_data = df.context.tolist()


#Training
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
model = Word2Vec(train_data, size=100, window=3, 
				min_count=2, workers=4, sg=1)


# Test
import numpy as np

key = "mạng"
result = np.array(model.wv.similar_by_word(key))
print("key: ", key, "\n", result)



# visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from importlib import reload
reload(sys)
words_np = []
words_label = []
for word in model.wv.vocab.keys():
    words_np.append(model.wv[word])
    words_label.append(word)

pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)

plt.rcParams["figure.figsize"] = (20,20)
for index,vec in enumerate(reduced):
    if index <200:
        x,y=vec[0],vec[1]
        plt.scatter(x,y)
        plt.annotate(words_label[index],xy=(x,y))
plt.show()