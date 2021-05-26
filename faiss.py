# Faiss相似向量检索加速

import numpy as np
import faiss
import time
import pandas as pd
start = time.time()
xb = np.load('newstitle.npy', allow_pickle=True)
xq = np.load('xq1.npy', allow_pickle=True)
print('加载标题时间：',time.time() - start)
print(xb.shape)

total_news = pd.read_csv('totalnews.csv')
title = total_news['title']
d = 768

nlist = 100
quantizer  = faiss.IndexFlatL2(d)   # build the index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

start1 = time.time()
assert not index.is_trained
index.train(xb)
assert index.is_trained
print('训练时间：',time.time() - start1)
index.add(xb) 

k = 4                          # we want to see 4 nearest neighbors
start2 = time.time()
D, I = index.search(xq, k)     # actual search
print(I)                   # neighbors of the 5 first queries
print('搜索时间：',time.time() - start2)
for i in I:
    print([title[j] for j in i])
print('总耗时：',time.time() - start)
