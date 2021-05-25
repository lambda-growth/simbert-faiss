
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
import time
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

maxlen = 32
# bert配置
config_path = 'chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])
# text = u'京津冀秋冬防'
# token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
# print(token_ids,segment_ids)


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def news_extract(news):
    X, S = [], []
    for title in news:
        x, s = tokenizer.encode(title, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    encode_title_ls = np.array(Z)
    # np.save('newstitle.npy',encode_title_ls)
    return Z

encode_title = np.load('newstitle.npy', allow_pickle=True)



def synonyms(text,title, k=20):
    # 加载预选词向量
    # encode_title_ls = encode_title.tolist()
    start1 = time.time()
    Z = news_extract(text)
    print('编码时间',time.time() - start1)
    # Z = np.vstack((Z,encode_title))
    start2 = int(round(time.time() * 1000))
    # Z /= (Z**2).sum(axis=1, keepdims=True)**0.5 #结论：L2范数归一化就是向量中每个元素除以向量的L2范数
    ls = np.dot(encode_title, -Z[0])
    print('矩阵乘法时间：{}ms'.format(int(round(time.time() * 1000)) - start2))
    start3 = int(round(time.time() * 1000))
    argsort = ls.argsort() #将X中的元素从小到大排序
    print('排序时间：{}ms'.format(int(round(time.time() * 1000)) - start3))
    return [title[i] for i in argsort[:k]]


from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist



# kmeans=KMeans(n_clusters=5)
# encode_title = np.load('newstitle.npy', allow_pickle=True)
# centers = kmeans.fit(encode_title[:1000])
# print(centers)

# print(kmeans.predict(Z))
# print(kmeans.labels_== kmeans.predict(Z).values)
#
total_news = pd.read_csv('totalnews.csv')
title = total_news['title']
# 方法一获取

while True:
    # 获取预选词向量

    # news_extract(title)
    # text = input('input:')
    start = time.time()

    text = ['京津冀秋冬防']
    a = synonyms([text],title, k=20)
    xq = news_extract(text)
    np.save('xq1.npy', xq)
    during = time.time() - start
    print('time used:', during) #3.34700
    print(a)










