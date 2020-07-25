#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().system(' jupyter nbconvert --to script recsystools.ipynb')


# In[1]:


import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import sys
import scipy
import bottleneck as bn
import math
import os


# class RecsysMetrics():
#     def __init__(self):
#         pass

# In[ ]:


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    
    # bn.argpartition
    # kth 번째까지 등장하는 원소들이 리스트 내부에서 가장 작은 kth번째 원소들이도록 partition해주는 인덱스 리스트를 출력해줌.
    # 여기서는 -를 취해줬으므로, k번째까지 등장하는 원소들이 리스트 내부에서 가장 큰 100번째 원소들이도록 partition해주는 인덱스 리스트를 출력.
    
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]
    
    # np.argsort -> sorting한 리스트의 arg를 뱉어냄. 
    # -를 붙여줌으로써 내림차순으로 정리(높은 놈이 위에)
    idx_part = np.argsort(-topk_part, axis=1)
    
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    
    DCG_filter=(heldout_batch[np.arange(batch_users)[:, np.newaxis],idx_topk].toarray()>0)
    
    DCG = (DCG_filter * tp).sum(axis=1)
    
    
    # sparse matrix 내에서, 고객의 총 interaction 수와 k 중 더 작은 것을 골라서 IDCG 계산
    
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    
    return DCG / IDCG


# In[2]:


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


# n_items_u는 각 유저별로 interact한 내역을 모아놓은 정보인 group의 길이(len)
# 이 길이가 5 이상일 경우, test_prop에 해당하는 비율을 np.random.choice로 샘플링해 idx로 저장
# 이 idx에 비해당 하는 interaction을 training data로 저장하고,
# 그 외는 test data로 저장함.

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('CustomerID')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te



# 해당 칼럼의 그룹 사이즈를 출력
def get_count(tp, id_):
    count_groupbyid = tp[[id_]].groupby(id_, as_index=False)
    count = count_groupbyid.size()
    return count

# 5개 미만으로 구매한 고객과 5번 미만으로 판매된 상품은 배제됨.
def filter_triplets(data, min_user_count=5, min_item_count=5):
    
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    
    if min_item_count > 0:
        itemcount = get_count(data, 'Description')
        data = data[data['Description'].isin(itemcount.index[itemcount >= min_item_count])]
    
    # Only keep the triplets for users who clicked on at least min_user_count items
    # After doing this, some of the items will have less than min_user_count users, but should only be a small proportion
    
    if min_user_count > 0:
        usercount = get_count(data, 'CustomerID')
        data = data[data['CustomerID'].isin(usercount.index[usercount >= min_user_count])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(data, 'CustomerID'), get_count(data, 'Description') 
    return data, usercount, itemcount


# In[ ]:




