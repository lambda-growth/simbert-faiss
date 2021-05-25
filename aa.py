import numpy as np
import faiss
import time
import pandas as pd
start = time.time()
xb = np.load('newstitle.npy', allow_pickle=True)
xq = np.load('xq.npy', allow_pickle=True)
print('加载标题时间：',time.time() - start)
print(xb.shape)

total_news = pd.read_csv('totalnews.csv')
title = total_news['title']
d = 768


index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
start2 = time.time()
D, I = index.search(xq, k)     # actual search
print(I)                   # neighbors of the 5 first queries
print('搜索时间：',time.time() - start2)
for i in I:
    print([title[j] for j in i])
print('总耗时：',time.time() - start)
