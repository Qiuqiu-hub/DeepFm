#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split


# In[2]:

def get_feature(data):
    config_column = pickle.load(open('./dic','rb'))
    feature_size = config_column['feature_size_']
    shape = data.shape
    feature_index = data.copy(deep = True)
    feature_value = data.copy(deep = True)
    for column in onehot_list:
        labelencoder = config_column[column]['labelencoder']
        feature = config_column[column]['feature']
        tran_x = [x + feature for x in labelencoder.transform(data[column])]
        data[column] = tran_x
        feature_index[column] = tran_x
        feature_value[column] = 1
    for column in continue_list:
        max_, min_ = config_column[column]['max_'],config_column[column]['min_']
        tran_x_continue = [(x - min_)/(max_ - min_) for x in data[column]]
        feature_index[column] = config_column[column]['feature']
        data[column] = tran_x_continue
        feature_value[column] = tran_x_continue
    
    return data, feature_index, feature_value, feature_size

def data_dataset_process(data, name = None, feature_size = 18, shuffle = True, batch_size = 512):
    
   
    index_,value_,label = data[:,:feature_size],data[:,feature_size:-1],data[:,-1]
    print('%s 数据的index,value,label的维度：',index_.shape,value_.shape,label.shape)
    #if len(index_[0]) != feature_size or len(value_[0]) != feature_size or len(label[0]) != 1:
     #   raise "数据分割错误"
    
    
    ds = tf.data.Dataset.from_tensor_slices(((index_,value_), label))
    ds = ds.shuffle(buffer_size=len(ds))
    ds = ds.batch(batch_size)
    return ds

def data_pre_split(feature_index, feature_value, rate = 0.15, del_list = ['fk_student','is_trans']):
    for column in del_list:
        index_ = feature_index.pop(column)
        value_ = feature_value.pop(column)
        if column == 'is_trans':
            label = index_
            
    index_ = feature_index.values
    value_ = feature_value.values
    label = label.values
    print(index_.shape, value_.shape,label.shape, len(index_), len(value_), len(label))
    if index_.shape != value_.shape or len(index_) != len(value_) != len(label):
        raise "数据维度不同！"
    
    concat = np.hstack((index_,value_,np.reshape(label,newshape = (-1,1))))
    train,test = train_test_split(concat, test_size = rate)
    train,val = train_test_split(train, test_size = rate)    
    
    return train,val,test


# In[ ]:

'''
#连续特征归一化
continue_list = ['total_real_price','province_cnt','city_cnt','city_trans_cnt','province_trans_cnt','city_cvr','province_cvr']
onehot_list = ['channel_id','mode','device','source_type','city_id','hotpot_type','province_id','sex','city_level',
              'platform_name','register_channel']

data = pd.read_csv('./features_label_data_deepfm.csv',',')
data, feature_index, feature_value, feature_size = get_feature(data)

feature_index.to_csv('./feature_index.csv',index = False)
feature_value.to_csv('./feature_value.csv',index = False)
#data = pd.concat([feature_index, feature_value], axis = 1, ignore_index = False)

#del_list = ['fk_student','is_trans']
#for column in del_list:
    #label = data.pop(column)
    #label = label.T.drop_duplicates().T
    #data[column] = label.values


'''
# In[ ]:

feature_index = pd.read_csv('./feature_index.csv', sep = ',')
feature_value = pd.read_csv('./feature_value.csv', sep = ',')

# In[ ]:

print('--------------data_to_dataset------------------')

train, val, test = data_pre_split(feature_index,feature_value, 0.15)
train = data_dataset_process(train, name = 'train')
test = data_dataset_process(test, name = 'test', shuffle= False)
val = data_dataset_process(val, name = 'val')

print('----------------dataset 分割结束----------------')
print("--**--**--"*16+'\n')


class DeepFM(tf.keras.layers.Layer):
    """
        init: 初始化参数
        build: 定义权重
        call: 层的功能与逻辑
        compute_output_shape: 推断输出模型维度
    """
 
    def __init__(self, embedding_size = None, dense_embed = None ,
                 regularizers ='l2',regularizers_ = 0.1, feature_size = None): 
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.dense_embed = dense_embed
        
        if regularizers == 'l2':
            self.kernel_regularize = tf.keras.regularizers.l2(regularizers_)
        else:
            self.kernel_regularize =tf.keras.regularizers.l1(regularizers_)
            
        self.activation = tf.keras.layers.Activation('relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
 
        self.bn2 = tf.keras.layers.BatchNormalization()

        super(DeepFM,self).__init__()
 
    # 定义模型初始化 根据特征数目
    def build(self, input_shape):
        # 特征隐向量矩阵
       
        
        self.feature_weight = self.add_weight(name='kernel',
                                      shape=(self.feature_size, self.embedding_size),
                                      initializer='glorot_normal',
                                      regularizer=self.kernel_regularize,
                                      trainable=True)
        
        self.first_weight = self.add_weight(name='kernel_fm',
                                      shape=(self.feature_size, ),
                                      initializer='glorot_normal',
                                      regularizer=self.kernel_regularize,
                                      trainable=True)

        self.first_bias = self.add_weight(name = 'firsr_bias',
                                          shape = (1,),
                                          initializer = 'glorot_normal',
                                          trainable = True)
 
        # DNN Dense1
        self.dense1 = self.add_weight(name='dense1',
                                      shape=(input_shape[-1] * self.embedding_size, self.dense_embed),
                                      initializer='glorot_normal',
                                      trainable=True)
 
        # DNN Bias1
        self.bias1 = self.add_weight(name='bias1',
                                     shape=(self.dense_embed, ),
                                     initializer='glorot_normal',
                                     trainable=True)
 
        # DNN Dense2
        self.dense2 = self.add_weight(name='dense2',
                                      shape=(self.dense_embed, self.dense_embed//2),
                                      initializer='glorot_normal',
                                      trainable=True)
 
        # DNN Bias2
        self.bias2 = self.add_weight(name='bias2',
                                     shape=(self.dense_embed//2, ),
                                     initializer='glorot_normal',
                                     trainable=True)
         
        # DNN Dense3
        self.dense3 = self.add_weight(name='dense3',
                                      shape=(self.dense_embed//2, self.dense_embed//4),
                                      initializer='glorot_normal',
                                      trainable=True)
 
        # DNN Bias3
        self.bias3 = self.add_weight(name='bias3',
                                     shape=(self.dense_embed//4, ),
                                     initializer='glorot_normal',
                                     trainable=True)
        
        # DNN Dense4
        self.dense4 = self.add_weight(name='dense4',
                                      shape=(self.dense_embed//4, 8),
                                      initializer='glorot_normal',
                                      trainable=True)
 
        
        self.bias4 = self.add_weight(name='bias4',
                                     shape=(8, ),
                                     initializer='glorot_normal',
                                      trainable=True)
        self.dense5 = self.add_weight(name = 'outputdense',
                                      shape = (10,1),
                                      initializer = 'glorot_normal',
                                      trainable = True) 
        
        super(DeepFM, self).build(input_shape)  # Be sure to call this at the end
 
    def call(self, inputs, **kwargs):
        # input 为多个样本的稀疏特征表示
        print('call_inputs.shape:',inputs.shape)
        feature_index = tf.reshape(tf.slice(inputs, [0,0,0],[-1,1,-1]),[-1,18])
        feature_index = tf.cast(feature_index, dtype = 'int32')
        feature_value = tf.slice(inputs,[0,1,0],[-1,1,-1])
        
        # LR
        first_order = self.get_first_order(feature_index)
        print('first_order:',first_order.shape)#1,1 -- none,
        # FM
        seconder_order = self.get_second_order(feature_index, feature_value)
        print('seconder_order:',seconder_order.shape)
        # DNN
        deep_order = self.get_deep_order(feature_index, feature_value)
        print(deep_order)# tenor none 1,2304  --- none,2304
        activation_1 = self.activation(self.bn1(tf.matmul(deep_order, self.dense1) + self.bias1))
        print(activation_1.shape)#1.64
        activation_2 = self.activation(tf.matmul(activation_1, self.dense2) + self.bias2)
        print(activation_2.shape)#1,32
        activation_3 = self.activation(self.bn2(tf.matmul(activation_2, self.dense3) + self.bias3))
        print(activation_3.shape)#1,16
        activation_4 = self.activation(tf.matmul(activation_3, self.dense4) + self.bias4)
        print('aaaaaaaaaaac:',activation_4.shape) #none,8
        # Concat
        concat = tf.stack([first_order,seconder_order],axis = -1)
        print(concat.shape)
        concat_order = tf.concat([concat, activation_4],axis = -1)
        print(concat_order.shape)
        #out = tf.sigmoid(tf.matmul(concat_order, self.dense5))
        #print(out.shape)
        return concat_order


    def compute_output_shape(self, input_shape):
            return input_shape(0)

    
    # LR线性部分
    def get_first_order(self,inp):
        #feature_index = tf.reshape(tf.slice(inputs, [0,0,0],[-1,1,-1]),[-1,18])
        #feature_index = tf.cast(feature_index, dtype = 'int32')
        #feature_value = tf.reshape(tf.slice(inputs,[0,1,0],[-1,1,-1]),[-1,18])
        first_weight = tf.nn.embedding_lookup(self.first_weight,inp)
        first_ = first_weight * feature_value
        print(K.sum(first_,axis = -1).shape)
        y_fm_fir = tf.add(tf.reduce_sum(first_,axis = -1), self.first_bias)
        return y_fm_fir


    # FM二阶交叉部分
    def get_second_order(self,inp1, inp2):
        
        print('-----------get_second_order_____________')
        #print(feature_value.shape, feature_index.shape)
        second_weight = tf.nn.embedding_lookup(self.feature_weight, inp1)
        print('second_weight_lookup:',second_weight.shape)
        second_weight = tf.transpose(inp2,[0,-1, -2]) * second_weight
        print('second_weight_transpose:',second_weight.shape)
        left = K.square(K.sum(second_weight,axis = 2))
        print('left',left.shape)
        right = K.sum(K.square(second_weight), axis = 2)
        print('right:',right.shape)
        y_fm_secon = 0.5 * tf.reduce_sum(tf.subtract(left, right),axis = -1)
        print('out',y_fm_secon.shape)
        print('------------get-second_order______________end__________-')
        return y_fm_secon

    # DNN高阶交叉部分
    def get_deep_order(self,inp1, inp2):
        # FM Args Shape: Feature_num * (K + 1)
        # Embedding Shape: Samples_num * Feature_num * K
        #feature_index = tf.slice(inputs, [0,0,0],[-1,1,-1])
        #feature_index = tf.cast(feature_index, dtype = 'int32')
        #feature_value = tf.slice(inputs,[0,1,0],[-1,1,-1])
        print('-------------get_deep_order____________________')
        feature_index = tf.squeeze(inp1, axis = 1)
        #print('feature_index:',feature_index.shape)

        #print('feature_value:',feature_value.shape)

        second_weight = tf.nn.embedding_lookup(weight, inp1)
        print('second_weight_lookup:',second_weight.shape) 
        second_weight = tf.transpose(inp2,[0,-1, -2]) * second_weight
        print('second_weight_transpose_to:',second_weight.shape) 

        # Flatten Shape: Samples_num * (Feature_num * K)
        embedding_flatten = tf.keras.layers.Flatten()(second_weight)
        print('embedding_flatten:',embedding_flatten.shape)
        print('--------get_deep_order________-end----------------------')
        return embedding_flatten


# In[ ]:


class FocalLoss(tf.keras.losses.Loss):
 
    def __init__(self,gamma=2.0,alpha=0.25, **kwargs):
        super(FocalLoss,self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
 
    def call(self,y_true,y_pred):
        y_true = tf.reshape(y_true,[-1,1]) 
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = K.sum(-tf.reduce_sum(self.alpha * tf.math.pow(1. - pt_1, self.gamma) * tf.math.log(1e-07+pt_1)-(1-self.alpha) * tf.math.pow( pt_0, self.gamma) * tf.math.log(1. - pt_0 + 1e-07),axis = -1),axis = -1)
        print('loss____:',loss.shape)
        return loss


# In[ ]:


# 构建模型
input_index = tf.keras.layers.Input(shape=(18,), name='input_index', dtype='float32')
input_value = tf.keras.layers.Input(shape=(18,), name='input_value', dtype='float32')
deepFm_layer = DeepFM(embedding_size = 128, dense_embed = 64 ,regularizers ='l2',regularizers_ = 0.1, feature_size = 666)([input_index,input_value])
#print(deepFm_layer.summary())

#reshape_deepFm = tf.keras.layers.Reshape((None,10),name = 'reshape')(deepFm_layer)

result = tf.keras.layers.Dense(1, activation='sigmoid', name='sigmoid')(deepFm_layer)
deepFm_model = tf.keras.Model([input_index, input_value], result)
 
# 模型编译
deepFm_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.01),
                         loss=tf.keras.losses.binary_crossentropy,
                         metrics=['accuracy','AUC'])
print('----------------------------deepFm_model.summary--------------------')
print(deepFm_model.summary())


# In[ ]:


logdir = './callback'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
print("--**--**--"*16+'\n')
print('开始训练：\n')
deepFm_model.fit(train, epochs = 20, verbose = 2, validation_data = val ,validation_freq = 3,
         callbacks =[tf.keras.callbacks.TensorBoard(logdir),
                     keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
                     keras.callbacks.EarlyStopping(monitor = 'val_loss',patience=2, min_delta=1e-3)])

