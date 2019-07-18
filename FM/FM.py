# encoding: utf-8
'''
@author: ltxu
@file: FM.py
@time: 2019/7/17 下午3:16
@desc:
'''
 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
 
cols = ['user','item','rating','timestamp']
train = pd.read_csv('ml-100k/ua.base', sep='\t',names=cols)
data = pd.read_csv('ml-100k/u.data', sep='\t',names=cols)
test = pd.read_csv('ml-100k/ua.test', sep='\t',names=cols)
 
n_user = len(data['user'].unique())
n_item = len(data['item'].unique())
y_train = train['rating'].values
y_test = test['rating'].values
 
train = train[['user','item']]
test = test[['user', 'item']]
# train.head()
 
ohr = OneHotEncoder(categories=[range(1,n_user+1), range(1,n_item+1)],sparse=False,dtype=np.int)
 
X_train = ohr.fit_transform(train)
 
# X_train.shape
 
n_feature = X_train.shape[1]
 
k = 10
 
X_test = ohr.transform(test)
 
# X_test.shape
 
w0 = tf.Variable(initial_value=tf.truncated_normal(shape=[1]), name='w0')
w = tf.Variable(initial_value=tf.truncated_normal(shape=[n_feature]), name='w')
 
# w.shape[0]
 
V = tf.Variable(initial_value=tf.truncated_normal(shape=[k, n_feature]), name='V')
 
X = tf.placeholder(dtype='float',shape=[None, n_feature], name="X")
y = tf.placeholder(dtype='float', shape=[None, 1], name= 'y')
 
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(X, w),axis=1,keepdims=True))
 
pair_interactions = 1/2 * tf.reduce_sum(
    tf.square(tf.matmul(X, V, transpose_b=True)) 
    - tf.matmul(tf.square(X), tf.square(V), transpose_b=True),
    axis=1, keepdims=True)
 
y_hat = linear_terms + pair_interactions
 
error = tf.reduce_mean(tf.square(y - y_hat))
 
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
 
l2_normal = lambda_w * tf.reduce_sum(tf.square(w)) + lambda_v*tf.reduce_sum(tf.square(V))
 
loss = error + l2_normal
 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
 
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)
 
epochs = 100
batch_size = 1000
 
# sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.summary.scalar(name='loss', tensor=loss)
merged = tf.summary.merge_all()

#train
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logdir='logs', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])

        for X_batch, y_batch in batcher(X_train[perm], y_train[perm], batch_size):

            s_time = time.time()
            epoch_loss,merged_value, _ = sess.run([loss, merged, train_op], feed_dict={X:X_batch.reshape(-1,n_feature), y:y_batch.reshape(-1,1)})
            print('epoch{}_loss: {}, epoch_running_time: {}'.format(epoch, epoch_loss, time.time()-s_time))
        if epoch%20 == 0:
            saver.save(sess, 'models/fm', global_step=epoch)
        train_writer.add_summary(merged_value, global_step=epoch)
    train_writer.close()
#test
with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    else:
        print('no model')
        exit(1)
    test_error, y_test_hat = sess.run([error, y_hat], feed_dict={X:X_test.reshape(-1,n_feature), y: y_test.reshape(-1,1)})
    # print(y_test_hat.shape)
    # mmse = tf.sqrt(tf.reduce_mean(tf.square(y_test_hat[:,0]-y_test)))
    print('mmse: ', mean_squared_error(y_test, y_test_hat[:,0]))
    print('test_error', test_error)