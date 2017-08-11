import tensorflow as tf

features = tf.constant([[1, 2, 3, 4], [6, 7, 8, 9]])
vector = tf.contrib.layers.embed_sequence(features, vocab_size=10, embed_dim=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(vector))
