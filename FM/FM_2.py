# encoding: utf-8
'''
@author: ltxu
@file: FM_2.py
@time: 2019/7/18 下午8:24
@desc: 使用user item 对应的属性，不再将user_id item_id 进行onehot
'''
# %%
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

# %%
# 处理user特征
u_user = pd.read_csv('ml-100k/u.user', sep='|', names=['u_id', 'age', 'gender', 'occupation', 'zip_code'])
encode = OrdinalEncoder(dtype=np.int).fit_transform(u_user[['gender', 'occupation', 'zip_code']])
u_user = pd.DataFrame(np.c_[u_user[['u_id', 'age']].values, encode],
                      columns=['u_id', 'age', 'gender', 'occupation', 'zip_code'])
u_user.head()
# %%

# %%
# #user_info_dict key:u_id, value:list()
# u_info_dict = {}
# for i in range(len(u_user)):
#     row = u_user.iloc[i]
#     u_info_dict[row['u_id']] = row[['age', 'gender', 'occupation', 'zip_code']].values
# %%
# 处理item特征
names = '''
m_id | m_title | release_date | video_release_date |
              IMDb_URL | unknown | Action | Adventure | Animation |
              Children | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western
              '''
names = [name.strip() for name in names.split('|')]
u_item = pd.read_csv('ml-100k/u.item', encoding='iso-8859-1', sep='|', names=names)
u_item.drop(['m_title', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)
u_item['release_date'].fillna(method='ffill', inplace=True)


# %%
def get_year(x):
    year_str = x[-4:]
    return int(year_str)


u_item['release_year'] = u_item['release_date'].apply(get_year)
u_item['release_year'] = u_item['release_year'] - u_item['release_year'].min()
u_item.drop('release_date', axis=1, inplace=True)
u_item.head()
# %%
cols = ['user', 'item', 'rating', 'timestamp']
data = pd.read_csv('ml-100k/u.data', sep='\t', names=cols)
train = pd.read_csv('ml-100k/ua.base', sep='\t', names=cols)
test = pd.read_csv('ml-100k/ua.test', sep='\t', names=cols)

n_user = len(data['user'].unique())
n_item = len(data['item'].unique())
y_train = train['rating'].values
y_test = test['rating'].values
train.head()
train = train[['user', 'item']]
test = test[['user', 'item']]
# %%
# item_info_dict = {}
# value_col = [
#     'release_year',
#  'unknown',
#  'Action',
#  'Adventure',
#  'Animation',
#  'Children',
#  'Comedy',
#  'Crime',
#  'Documentary',
#  'Drama',
#  'Fantasy',
#  'Film-Noir',
#  'Horror',
#  'Musical',
#  'Mystery',
#  'Romance',
#  'Sci-Fi',
#  'Thriller',
#  'War',
#  'Western',
#  ]
#
# for i in range(len(u_item)):
#     row = u_item.iloc[i]
#     item_info_dict[row['m_id']] = row[value_col].values
# %%
# 组合数据
train = pd.merge(train, u_user, how='left', left_on='user', right_on='u_id')
train.drop('u_id', axis=1, inplace=True)
train = pd.merge(train, u_item, how='left', left_on='item', right_on='m_id')
train.drop('m_id', axis=1, inplace=True)
train.drop(['user', 'item'], axis=1, inplace=True)
# %%
test = pd.merge(test, u_user, how='left', left_on='user', right_on='u_id')
test.drop('u_id', axis=1, inplace=True)
test = pd.merge(test, u_item, how='left', left_on='item', right_on='m_id')
test.drop('m_id', axis=1, inplace=True)
test.drop(['user', 'item'], axis=1, inplace=True)

# %%
ct = ColumnTransformer([
                        # ('u_i_onehot',OneHotEncoder(categories=[range(1, n_user + 1), range(1, n_item + 1)], sparse=False,dtype=np.int), ['user', 'item']),
                        ('gender_onehot', OneHotEncoder(dtype=np.int, sparse=False),['gender', 'occupation', 'zip_code'])
                        ],
                       remainder='passthrough')
ct.fit(train)
X_train = ct.transform(train)
X_test = ct.transform(test)

# %%
# 特征维度与V的维度
n_feature = X_train.shape[1]
k = 10
# %%
# 定义权重
w0 = tf.Variable(initial_value=tf.truncated_normal(shape=[1]), name='w0')
w = tf.Variable(initial_value=tf.truncated_normal(shape=[n_feature]), name='w')
V = tf.Variable(initial_value=tf.truncated_normal(shape=[k, n_feature]), name='V')
# %%
X = tf.placeholder(dtype='float', shape=[None, n_feature], name="X")
y = tf.placeholder(dtype='float', shape=[None, 1], name='y')
# %%
# 公式
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(X, w), axis=1, keepdims=True))
pair_interactions = 1 / 2 * tf.reduce_sum(
    tf.square(tf.matmul(X, V, transpose_b=True))
    - tf.matmul(tf.square(X), tf.square(V), transpose_b=True),
    axis=1, keepdims=True)
y_hat = linear_terms + pair_interactions

# %%
y_hat = linear_terms + pair_interactions

error = tf.reduce_mean(tf.square(y - y_hat))

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_normal = lambda_w * tf.reduce_sum(tf.square(w)) + lambda_v * tf.reduce_sum(tf.square(V))

loss = error + l2_normal
# %%
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# %%
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


# %%
epochs = 1000
batch_size = 1000
# %%
loss_scalar = tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logdir='FM_2_logs', graph=sess.graph)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='FM_2_models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
        epoch_start = 500
    else:
        epoch_start = 1
    for epoch in range(epoch_start, epochs + 1):
        s_time = time.time()
        perm = np.random.permutation(X_train.shape[0])
        for X_batch, y_batch in batcher(X_train[perm], y_train[perm], batch_size):
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={X: X_batch.reshape(-1, n_feature), y: y_batch.reshape(-1, 1)})
        print('epoch{}_loss: {}, epoch_running_time: {}'.format(epoch, loss_value, time.time() - s_time))

        if epoch % 20 == 0:
            saver.save(sess, save_path='FM_2_models/fm', global_step=epoch)
            merged_value = sess.run(merged, feed_dict={X: X_batch.reshape(-1, n_feature), y: y_batch.reshape(-1, 1)})
            train_writer.add_summary(merged_value, global_step=epoch)

            # test
            error_test, y_test_pred = sess.run([error, y_hat],
                                               feed_dict={X: X_test.reshape(-1, n_feature), y: y_test.reshape(-1, 1)})
            mmse = mean_squared_error(y_test, y_test_pred)
            print('loss_test: {}, mmse: {}'.format(error_test, mmse))

    train_writer.close()
# %%
# test
with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='FM_2_models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    else:
        print('no model')
        exit(1)
    test_error, y_test_hat = sess.run([error, y_hat], feed_dict={X:X_test.reshape(-1,n_feature), y: y_test.reshape(-1,1)})
    # test_error, y_test_hat = sess.run([error, y_hat], feed_dict={X:X_test.reshape(-1,n_feature), y: y_test.reshape(-1,1)})

    # print(y_test_hat.shape)
    # mmse = tf.sqrt(tf.reduce_mean(tf.square(y_test_hat[:,0]-y_test)))
    print('mmse: ', mean_squared_error(y_test, y_test_hat[:,0]))
    print('test_error', test_error)
# %%


