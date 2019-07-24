# encoding: utf-8
'''
@author: ltxu
@file: DeepFM_3.py
@time: 2019/7/24 上午10:09
@desc:
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import datetime
import joblib

#加入正则化

# %%
data_merged = pd.read_csv('data_merged.csv')
# %%
# %%
split_timestamp = 888721510
# split_time = datetime.datetime.utcfromtimestamp(split_timestamp)
train = data_merged[data_merged.timestamp <= split_timestamp]
y_train = train.like.values
train = train.drop(['user', 'item', 'timestamp', 'datetime', 'rating', 'like'], axis=1)
# %%
test = data_merged[data_merged.timestamp > split_timestamp]
y_test = test.like.values
test = test.drop(['user', 'item', 'timestamp', 'datetime', 'rating', 'like'], axis=1)

# %%
num_features = ['age', 'release_year']
ignore_features = ['user', 'item', 'timestamp', 'datetime', 'like', 'rating']
feature_dict = {}
count = 0
for col in data_merged.columns:
    if col in ignore_features:
        continue
    elif col in num_features:
        feature_dict[col] = count
        count += 1
    else:
        length = len(data_merged[col].unique())
        feature_dict[col] = dict(zip(data_merged[col].unique(), range(count, count + length)))
        count += length
# %%
total_feature = count
# %%
train_feature_index = train.copy()
train_feature_value = train.copy()
for col in train.columns:
    if col in num_features:
        train_feature_index[col] = [feature_dict[col]] * len(train)
    elif col in ignore_features:
        continue
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1
test_feature_index = test.copy()
test_feature_value = test.copy()
for col in test.columns:
    if col in num_features:
        test_feature_index[col] = [feature_dict[col]] * len(test)
    elif col in ignore_features:
        continue
    else:
        test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
        test_feature_value[col] = 1
# %%
"""模型参数"""
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 5,
    # "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    # "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": 'relu',  # or tanh
    "epoch": 1000,
    # "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer": "adam",
    # "batch_norm": 1,
    # "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    # "verbose": True,
    # "eval_metric": 'gini_norm',
    # "random_seed": 42
}
dfm_params['feature_size'] = total_feature
dfm_params['field_size'] = len(train_feature_index.columns)
# %%
# feat_index = tf.placeholder(shape=[None, dfm_params['field_size']], name='feat_index')
# feat_value = tf.placeholder(shape=[None, dfm_params['field_size']], name='feat_index')
feat_index = tf.placeholder(dtype=np.int32, shape=[None, None], name='feat_index')  # why
feat_value = tf.placeholder(dtype=np.float32, shape=[None, None], name='feat_value')
label = tf.placeholder(dtype=np.float32, shape=[None, 1], name='label')
# %%
weights = dict()
weights['feature_embeddings'] = tf.Variable(
    initial_value=tf.truncated_normal(shape=[dfm_params['feature_size'], dfm_params['embedding_size']], mean=0.0,
                                      stddev=0.01), name='feature_embeddings')
if dfm_params['use_fm']:
    weights['feature_bias'] = tf.Variable(
        initial_value=tf.truncated_normal(shape=[dfm_params['feature_size'], 1], mean=0.0, stddev=1.0),
        name='feature_bias')  # 与fm相同

# deep net
if dfm_params['use_deep']:
    num_layer = len(dfm_params['deep_layers'])
    input_size = dfm_params['field_size'] * dfm_params['embedding_size']

    glorot = np.sqrt(2.0 / (input_size + dfm_params['deep_layers'][0]))
    weights['layer_0'] = tf.Variable(
        initial_value=tf.truncated_normal(shape=[input_size, dfm_params['deep_layers'][0]], mean=0.0, stddev=glorot),
        name='layer_0')
    weights['bias_0'] = tf.Variable(initial_value=tf.truncated_normal(shape=[1, dfm_params['deep_layers'][0]]),
                                    name='bias_0')  # forget
    for i in range(1, num_layer):
        glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][i - 1] + dfm_params['deep_layers'][i]))
        weights['layer_%d' % i] = tf.Variable(
            initial_value=tf.truncated_normal(shape=[dfm_params['deep_layers'][i - 1], dfm_params['deep_layers'][i]],
                                              mean=0.0, stddev=glorot), name='layer_%d' % i)
        weights['bias_%d' % i] = tf.Variable(
            initial_value=tf.truncated_normal(shape=[1, dfm_params['deep_layers'][i]], mean=0.0, stddev=glorot),
            name='bias_%d' % i)

# final concat projection layer
if dfm_params['use_fm'] and dfm_params['use_deep']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
elif dfm_params['use_fm']:
    input_size = dfm_params['field_size'] + dfm_params['embedding_size']
else:
    input_size = dfm_params['deep_layers'][-1]

glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(
    initial_value=tf.truncated_normal(shape=[input_size, 1], mean=0.0, stddev=glorot))
weights['concat_bias'] = tf.Variable(initial_value=tf.constant(0.01))  # why
# %% md
#### fm用onehot处理 deepfm用embedding处理，方式不同，但处理结果是相同的，
#### embedding_lookup取出的是值为1的feature，值为0的丢弃，onehot后矩阵相乘，同样不会计算值为0
#### 在电影类别中（‘unknown', 'Action',etc）0,1为分类，不是上述的值为0或1，，如电影在unknown这个field为0，
#### 那么说明此电影其属于"0"这个分类，这个分类值为1，"1"这个分类值为0，在look_up时，的确没有计算“1”这个分类
#### 明确'unknown', 'Action', 'Adventure', 'Animation',这些都是field，0,1是这个field的两个分类。
# %%
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)
reshaped_feat_value = tf.reshape(feat_value, shape=[-1, dfm_params['field_size'], 1])
embeddings = tf.multiply(embeddings, reshaped_feat_value)
# %%
if dfm_params['use_fm']:
    # fm 一次项
    fm_first_order = tf.nn.embedding_lookup(weights['feature_bias'], feat_index)
    fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order, reshaped_feat_value), 2)
    # %%
    # fm 二次项
    summed_features_emb = tf.reduce_sum(embeddings, 1)
    summed_features_emb_square = tf.square(summed_features_emb)

    feature_emb_square = tf.square(embeddings)
    feature_emb_square_sum = tf.reduce_sum(feature_emb_square, 1)
    fm_second_order = 0.5 * tf.subtract(summed_features_emb_square, feature_emb_square_sum)
# %%
# deep
if dfm_params['use_deep']:
    y_deep = tf.reshape(embeddings, shape=[-1, dfm_params['embedding_size'] * dfm_params['field_size']])
    for i in range(num_layer):
        y_deep = tf.matmul(y_deep, weights['layer_%d' % i])
        y_deep = tf.add(y_deep, weights['bias_%d' % i])
        if dfm_params['deep_layer_activation'] == 'relu':
            y_deep = tf.nn.relu(y_deep)
        elif dfm_params['deep_layer_activation'] == 'tanh':
            y_deep = tf.nn.tanh(y_deep)
# %%
# final layer
if dfm_params['use_fm'] and dfm_params['use_deep']:
    concat_input = tf.concat([fm_first_order, fm_second_order, y_deep], axis=1)
elif dfm_params['use_fm']:
    concat_input = tf.concat([fm_second_order, fm_first_order], axis=1)
else:
    concat_input = y_deep

out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input,weights['concat_projection']),weights['concat_bias']))
#%%
# def get_batch(Xi, Xv, y, batch_size, index):
#     start = index * batch_size
#     end = (index + 1) * batch_size
#     end = end if end < len(y) else len(y)
#     return Xi[start:end], Xv[start:end], y[start:end].reshape(-1,1)
# num_batch = train_feature_index.shape[0]//dfm_params['batch_size'] + 1
# %%

loss = tf.losses.log_loss(tf.reshape(label, [-1, 1]), out)

#l2 reg
for var in weights.values():
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
regularizer = tf.contrib.layers.l2_regularizer(dfm_params['l2_reg'])
reg_term = tf.contrib.layers.apply_regularization(regularizer)
loss_reg = loss + reg_term

loss_value_ = tf.placeholder(dtype=np.float32, name='loss_values_')
train_loss_summary = tf.summary.scalar(name='train_loss_reg', tensor=loss_value_)
test_loss_summary = tf.summary.scalar(name='test_loss', tensor=loss_value_)
auc_value_ = tf.placeholder(dtype=np.float32, name='auc_value_')
train_auc_summary = tf.summary.scalar(name='train_auc', tensor=auc_value_)
test_auc_summary = tf.summary.scalar(name='test_auc', tensor=auc_value_)
global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(global_step, 1, name='global_step_op')

optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'], beta1=0.9, beta2=0.999 ).minimize(loss_reg)
# %%
# merged = tf.summary.merge()
saver = tf.train.Saver()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir='dfm_3_log', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='dfm_3_models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        last_step = sess.run(global_step)
        # sess.run(tf.assign(global_step, last_step))
        print('get last_step:', last_step)
    else:
        last_step = 0
    # print('last_step: ', last_step)
    for i in range(last_step+1,dfm_params['epoch']+1):
        sess.run(optimizer, feed_dict={feat_index: train_feature_index,
                                                               feat_value: train_feature_value,
                                                               label: y_train.reshape(-1, 1)})
        # auc_value = roc_auc_score(y_train.reshape(-1,1), out_value)
        # print('epoch{}, loss: {}, auc: {}'.format(i, loss_value, auc_value))
        sess.run(increment_global_step)
        # print('global step: ', sess.run(global_step))
        if i%10==0:
            loss_value, out_value = sess.run([loss_reg, out], feed_dict={feat_index: train_feature_index,
                                                               feat_value: train_feature_value,
                                                               label: y_train.reshape(-1, 1)})
            auc_value = roc_auc_score(y_train.reshape(-1,1), out_value)
            print('epoch{}, loss: {}, auc: {}'.format(i, loss_value, auc_value))
            saver.save(sess, save_path='dfm_3_models/dfm', global_step=global_step)

            train_loss_summary_value, train_auc_summary_value = sess.run([train_loss_summary, train_auc_summary], feed_dict={loss_value_:loss_value, auc_value_:auc_value})
            writer.add_summary(train_auc_summary_value, global_step=i)
            writer.add_summary(train_loss_summary_value, global_step=i)

            loss_value, out_value = sess.run([loss, out], feed_dict={feat_index:test_feature_index, feat_value: test_feature_value, label:y_test.reshape(-1,1)})
            # print('test out_value:', out_value.shape)
            # print('test y_batch:', test_feature_index.shape)
            auc = roc_auc_score(y_test.reshape(-1,1), out_value)
            test_loss_summary_value = sess.run(test_loss_summary, feed_dict={loss_value_:loss_value})
            writer.add_summary(test_loss_summary_value, global_step= i)
            test_auc_summary_value = sess.run(test_auc_summary, feed_dict={auc_value_:auc})
            writer.add_summary( test_auc_summary_value, global_step= i)


writer.close()