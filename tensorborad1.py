import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layerxxx%s' % n_layer
    with tf.name_scope('layeryyy'):
        with tf.name_scope('weightsyyy'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w_name')
            tf.summary.histogram(layer_name + '/weightsxxx', Weights)
        with tf.name_scope('biasesyyy'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b_name')
            tf.summary.histogram(layer_name + '/biasesxxx', biases)
        with tf.name_scope('Wx_plus_byyy'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputsxxx', outputs)
        return outputs
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputsyyy'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)
with tf.name_scope('lossyyy'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    tf.summary.scalar('lossxxx', loss)
init = tf.global_variables_initializer()
with tf.name_scope('trainyyy'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
   if i % 50 == 0:
       rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
       writer.add_summary(rs, i)

