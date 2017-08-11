"""
7.17 learn
https://www.tensorflow.org/get_started/get_started
"""
import tensorflow as tf

sess = tf.Session()

node1 = tf.constant(3.)
node2 = tf.constant(4.)
print(node1,node2)

node3 = tf.add(node1,node2)
print(node3)
print("sess.run(node3)",sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node,{a:[1,20],b:[3,5]}))
add_and_triple = adder_node*3
print(sess.run(add_and_triple,{a:2,b:4}))



W = tf.Variable([4],dtype=tf.float32)
b = tf.Variable([-4],dtype=tf.float32)
x = tf.placeholder(tf.float32)
liner_model = W * x + b
init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(liner_model,{x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(liner_model-y)
loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss,{x:[1,2,3,4],y:[5,6,7,8]}))

# fixW = tf.assign(W, [1.])
# fixb = tf.assign(b, [4.])
# sess.run([fixW, fixb])
# print(sess.run(loss, {x:[1,2,3,4], y:[5,6,7,8]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss=loss)
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run([W,b]))