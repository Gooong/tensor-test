import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")
batch = mnist.test.next_batch(500)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
image = tf.reshape(x, shape=[-1, 28, 28])

cell = tf.contrib.rnn.LSTMCell(28)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
output, _ = tf.nn.dynamic_rnn(cell, image, dtype=tf.float32)
output = tf.reshape(output, [-1, 784])
logits = tf.contrib.layers.fully_connected(output, 10)

y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
loss = tf.contrib.losses.softmax_cross_entropy(logits, y_)
my_loss = tf.identity(loss)
optmizer = tf.contrib.layers.optimize_loss(my_loss, tf.contrib.framework.get_global_step(), optimizer='Adam',
                                           learning_rate=0.01)

correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        numbers, labels = mnist.train.next_batch(300)
        sess.run(optmizer, feed_dict={x: numbers, y_: labels})
        if i % 100 == 0:
            c = sess.run(my_loss, feed_dict={x: numbers, y_: labels})
            print("step: %d, training loss: %.2f" % (i, c))

            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))
    validation_batch = mnist.validation.next_batch(3000)
    validation_accuracy = accuracy.eval(feed_dict={x:validation_batch[0], y_:validation_batch[1]})
    print("finally, validation accuracy %g" % (validation_accuracy))