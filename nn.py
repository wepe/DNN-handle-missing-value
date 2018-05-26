import tensorflow as tf
from dataio import data_fill_zero,data_with_missing_value

n_input = 52
n_hidden_1 = 128
n_hidden_2 = 128
n_output = 1

X = tf.placeholder('float',[None,n_input])
Y = tf.placeholder('float',[None,n_output])
is_training = tf.placeholder(tf.bool)

weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'out':tf.Variable(tf.random_normal([n_hidden_2,n_output]))}

biases = {'h1':tf.Variable(tf.random_normal([n_hidden_1])),
          'h2':tf.Variable(tf.random_normal([n_hidden_2])),
          'out':tf.Variable(tf.random_normal([n_output]))}

layer1 = tf.add(tf.matmul(X,weights['h1']),biases['h1'])
layer1 = tf.layers.dropout(layer1, rate=0.5, training=is_training)
layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['h2'])
layer2 = tf.layers.dropout(layer2, rate=0.5, training=is_training)
logit = tf.add(tf.matmul(layer2,weights['out']),biases['out'])

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=Y)) + \
        tf.contrib.layers.l2_regularizer(0.001)(weights['h1']) + \
        tf.contrib.layers.l2_regularizer(0.001)(weights['h2']) + \
        tf.contrib.layers.l2_regularizer(0.001)(weights['out'])

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

prob = tf.sigmoid(logit)
auc,metric_op = tf.metrics.auc(labels=Y,predictions=prob)

best_auc = 0.0
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    train,(val_x,val_y) = data_fill_zero(batch_size=1024)
    print val_x.shape,val_y.shape
    for epoch in range(2000):
        for batch_x,batch_y in train():
            train_loss,_ = sess.run([loss,train_op],feed_dict={X:batch_x,Y:batch_y,is_training:True})

        val_auc = sess.run(metric_op,feed_dict={X:val_x,Y:val_y,is_training:False})
        print "epoch:{},val auc:{}".format(epoch,val_auc)
        best_auc = max(val_auc,best_auc)

print 'best auc:{}'.format(best_auc)

