#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split

import os 


# In[4]:


data = pd.read_csv('./features_label_data_deepfm.csv',',')

plt.figure(12)
i = 1
for column in ['total_real_price','province_cnt','city_cnt','province_trans_cnt']:
    
    plt.subplot(240+i)
    i += 1
    data[column].plot()
    plt.subplot(240+i)
    data[column].hist()
    i+=1


# ## 数据处理

# In[7]:


#选择保留的特征
selected_list = ['fk_student', 'channel_id', 'mode', 'device', 'source_type', 'city_id', 'hotpot_type','province_id',
                 'sex','city_level', 'total_real_price', 'province_cnt', 'city_cnt', 'city_trans_cnt',
                 'province_trans_cnt', 'city_cvr', 'province_cvr', 'is_trans','platform_name','register_channel']

#连续特征归一化
continue_list = ['total_real_price','province_cnt','city_cnt','city_trans_cnt','province_trans_cnt','city_cvr','province_cvr']
onehot_list = ['channel_id','mode','device','source_type','city_id','hotpot_type','province_id','sex','city_level',
              'platform_name','register_channel']


# In[8]:


#离散特征：onehot 记录labelencoder实例和对应的feature_size
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import pickle
dic = {}
dic_feature = {}
def one_hot_labelencoder(data):
    feature_size = 0
    for column in onehot_list:
        dic_ = {}
        i = 0
        tmp_data = data[column]
        lab = LabelEncoder()
        lab.fit(tmp_data)
        column_size = len(lab.classes_)
        for class_ in lab.classes_:
            dic_[class_] = i + feature_size
            i += 1
        dic[column] = {'labelencoder':lab,'feature':feature_size}
        dic_feature[column] = {'%s_numdic'%column:dic_,'feature':feature_size}
        data[column] = [x + feature_size for x in lab.transform(data[column])]
        feature_size += column_size
        dic['feature_size_'] = feature_size
        dic_feature['feature_size_'] = feature_size
    with open('./dic','wb') as f: 
        pickle.dump(dic,f)
        #读取 pickle.load(open('./dic','rb'))
    with open('./dic_feature.json','w') as f:
        json.dump(dic_feature, f)


data.to_csv('./deepfm_onehot_feature.csv',index = False)
data = pd.read_csv('./deepfm_onehot_feature.csv',',')
#连续特征处理： 归一化 记录均值和方差

def continue_feature(data):
    with open('./dic','rb') as file:
        dic = pickle.load(file)
    with open('./dic_feature.json','r') as ff:
        dic_feature = json.load(ff)
    feature_size = dic['feature_size_']
    for column in continue_list:
        mean_ = data[column].mean()
        max_ = data[column].max()
        min_ = data[column].min()
        dic[column] = {'max_':max_,'min_':min_,'mean_':mean_, 'feature':feature_size}
        dic_feature[column] = feature_size
        feature_size += 1
    dic['feature_size_'] = feature_size
    data[continue_list] = data[continue_list].apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    with open('./dic','wb') as f:
        pickle.dump(dic,f)
    with open('./dic_feature.json','w') as f:
        json.dump(dic_feature,f)


# In[12]:

continue_feature(data)
data.to_csv('./deep_fm_onehotcontinue.csv',index = False)
