
# coding: utf-8

# In[21]:

import tensorflow as tf
import operator
import sys
import os 

# Check file exists
if(len(sys.argv) < 2):
	print("Please enter filename")
	exit()

validate_filename = sys.argv[1]
print("Filename: ", validate_filename)
if not (os.path.isfile(validate_filename)):
        print("File not found")
        exit()

sess = tf.InteractiveSession()

coord = tf.train.Coordinator()

NUM_OF_CLASS = 50
IMG_X = 64
IMG_Y = 64


IS_TRAINING = False


# In[2]:



#filename_queue = tf.train.string_input_producer([
#    'train/gut_high (1).jpg',
#    'train/gut_high (2).jpg',
#    'train/nail_mid (1).jpg']) #  list of files to read

filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('train/*.jpg'))

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

#print(value)
#print(key)

#my_img = tf.placeholder(tf.zeros([IMG_X, IMG_Y , 3]), name="imgVar")
my_img = tf.reshape(tf.to_float(tf.image.decode_jpeg(value)), [IMG_X, IMG_Y, 3]) # use png or jpg decoder based on your files.

#adding key

#my_img = tf.pack([my_img, key])


#_label = [[1,0], [1,0], [0,1] ]
#label = tf.reshape(_label, [-1,  2])

##TEST

#with sess.as_default() as sess:
#    sess.run(tf.global_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
filename_before_slash = tf.sparse_tensor_to_dense(tf.string_split([key], delimiter='/'), default_value="x")[0][1]
filename_before_space = tf.sparse_tensor_to_dense(tf.string_split([filename_before_slash], delimiter=' '), default_value="x")[0][0]


hashed_index = tf.string_to_hash_bucket_strong(filename_before_space, NUM_OF_CLASS, [4758937974, 729479902])

label = tf.one_hot(hashed_index, NUM_OF_CLASS)
    
tf.summary.histogram("Label_hashed_index",hashed_index)
    

batch_size = 33
min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch(
    [my_img, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

#    print(sess.run( [label ,filename_before_space] ))
#    print(sess.run( [label ,filename_before_space] ))
#    print(sess.run( [label ,filename_before_space] ))
#    print(sess.run( [label ,filename_before_space] ))
#    print(sess.run( [label ,filename_before_space] ))
#    
#    #print(sess.run(my_img))
#    
#    coord.request_stop()
#    coord.join(threads)

#example_batch, label_batch = tf.train.shuffle_batch(
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
# 
#     # Start populating the filename queue.
#     print("A")
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     print("B")
#     for i in range(1): #length of your filename list
#         image = my_img.eval() #here is your image Tensor :) 
#     print("C")
#     print(image.shape)
#     #Image.show(Image.fromarray(np.asarray(image)))
# 
#     coord.request_stop()
#     coord.join(threads)


# In[3]:

# print("Downloading")
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)




# In[ ]:




# In[4]:

x = tf.placeholder(tf.float32, shape=[None, IMG_X * IMG_Y * 3], name="placeholderX")

#x = tf.Variable(tf.zeros([1, IMG_X * IMG_Y * 3]), name="placeholderX")

#x = tf.pack([tf.reshape(tf.to_float(my_img), [IMG_X * IMG_Y * 3]), tf.reshape(tf.to_float(my_img), [IMG_X * IMG_Y * 3]), tf.reshape(tf.to_float(my_img), [IMG_X * IMG_Y * 3])])
#y_ = tf.pack([label_dict[key], label_dict[key], label_dict[key]])

y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_CLASS], name="placeholderY")

#y = tf.pack([key, key, key])

# x = tf.reshape(example_batch, [-1, IMG_X * IMG_Y * 3])
# y_ = label_batch

tf.summary.histogram("example_batch",example_batch)
tf.summary.histogram("label_batch",y_)


# W = tf.Variable(tf.zeros([IMG_X * IMG_Y * 3, NUM_OF_CLASS]), name="variableWeight")
# w_hist = tf.summary.histogram("Weight_Hist", W)
# 
# b = tf.Variable(tf.zeros([NUM_OF_CLASS]), name="vaiableBias")


# In[5]:

# Y is x times weight + bias
#y = tf.matmul(x,W) + b



# for i in range(20):
#     batch = mnist.train.next_batch(10)
#     feed = {x: mnist.test.images, y_: mnist.test.labels}
#     result = sess.run([merged, train_step], feed_dict=feed)
#     writer.add_summary(result[0],i)




# In[6]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[7]:

#1st Layer
with tf.name_scope('layer_1') as scope:
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    # Mode 3 for colored
    x_image = tf.reshape(x, [-1, IMG_X, IMG_Y, 3]) 

    #Convolve Image by weight tensor, add bias, and apply Relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #Maxpool, reduces to 14 x 14
    h_pool1 = max_pool_2x2(h_conv1)


#2nd Layer
with tf.name_scope('layer_2') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #Maxpool, reduces to 7x7
    h_pool2 = max_pool_2x2(h_conv2)



# In[8]:


#Output layer
with tf.name_scope('layer_3') as scope:
    #Add 1024 Neuron for 7x7  image

    # 7 x 7: Resolution after pooling
    # 64: Num of features

    W_fc1 = weight_variable([(IMG_X //2 // 2) * (IMG_Y //2 //2) * 64, 1024])
    b_fc1 = bias_variable([1024])

    #Reshape to batch of vector
    h_pool2_flat = tf.reshape(h_pool2, [-1, (IMG_X //2 // 2) * (IMG_Y //2 //2) * 64 ])
    
    #Do Bias and Relu stuffs
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #Dropout to prevent overfit
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Readout

    W_fc2 = weight_variable([1024, NUM_OF_CLASS])
    b_fc2 = bias_variable([NUM_OF_CLASS])
    
    tf.summary.histogram("W_fc2", W_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    tf.summary.histogram("y_conv_argmax", tf.argmax(y_conv,1))
    
#Now using y_conv

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_), name="Entr")
cross_entropy_hist = tf.summary.scalar("Cross_Entropy", cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="Acc")
accuracy_hist = tf.summary.scalar("Accuracy", accuracy)

#How to train
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[ ]:




# In[9]:

#Load images
with sess.as_default() as sess:
    if IS_TRAINING:
      with tf.name_scope("LoadTestImg"):
              def load_test_img():
                  threads = tf.train.start_queue_runners(coord=coord)

                  train_img_arr, label_img_arr = sess.run([example_batch, label_batch])
                  return [train_img_arr, label_img_arr]

    with tf.name_scope("LoadValidateImg"):
        validate_queue = tf.train.string_input_producer([validate_filename])
        readerVal = tf.WholeFileReader()
        filename, dat = readerVal.read(validate_queue)
        val_img = tf.to_float(tf.image.decode_jpeg(dat)) # use png or jpg decoder based on your files.

        patches = tf.extract_image_patches([val_img],[1, 64, 64, 1], [1, 32, 32, 1], [1, 1, 1, 1], 'SAME')
        
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        threads2 = tf.train.start_queue_runners(coord=coord)
    #with sess.as_default() as sess:
        #sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #threads2 = tf.train.start_queue_runners(coord=coord)


# In[10]:

if IS_TRAINING:
    #Do Training
    with sess.as_default() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist_logs", sess.graph)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        #img = [sess.run(my_img) for i in range(3)]


        #_filename,img = sess.run([key, my_img])
        #filename = _filename.decode("utf-8") #Converts from bytes type
        #print(filename.upper()[0])

        #print(label_dict[filename])


        #expected_result = tf.reshape(label_dict[filename], [-1, 2]).eval()

        #for i in range(8):

        # load batch


        for i in range(200):        
            train_img_arr, label_img_arr = load_test_img()
            result, _ = sess.run([merged, train_step], feed_dict={
                x: tf.reshape(train_img_arr, [-1, IMG_X * IMG_Y * 3]).eval(),
                y_: label_img_arr,
                keep_prob: 0.5})
            writer.add_summary(result,i)
            train_accuracy = sess.run(accuracy, feed_dict={
                x: tf.reshape(train_img_arr, [-1, IMG_X * IMG_Y * 3]).eval(),
                y_: label_img_arr,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))


        # coord.request_stop()
        # coord.join(threads)
        #for i in range(1):
        #        batch = mnist.train.next_batch(500)
        #        if i%100 == 0:
        #            train_accuracy = sess.run(accuracy, feed_dict={
        #                x:batch[0], y_: batch[1], keep_prob: 1.0})
        #            print("step %d, training accuracy %g"%(i, train_accuracy))
        #            result, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})
        #            writer.add_summary(result,i)

else:
    print("TRAINING SKIPPED")


# In[11]:

# create saver object.
saver = tf.train.Saver()
if (IS_TRAINING):
    #Save variable
    with sess.as_default() as sess:
        saver.save(sess, '/tmp/skin_saves/vars.ckpt')
else:
    #Load variable
    with sess.as_default() as sess:
        saver.restore(sess, '/tmp/skin_saves/vars.ckpt')


# In[ ]:




# In[ ]:




# In[12]:

c = {}
classes = ['gut_low', 'Gut_mid', 'Gut_high', 'nail_low', 'nail_mid', 'nail_high', 'plaque_low', 'plaque_mid', 'plaque_high', 'pus_low', 'pus_high']
with sess.as_default() as sess:
    #print(sess.run(tf.argmax(y_conv,1), feed_dict={keep_prob: 1.0}))
    for clsName in classes:
        classIdx = sess.run([tf.argmax([label],1)], feed_dict={filename_before_space: clsName})[0][0]
        c[classIdx] = clsName


# In[13]:

with sess.as_default() as sess:
    fname, patches_arr = sess.run([filename,patches])

validate_accuracy = sess.run(tf.argmax(y_conv, 1), feed_dict={
    x: tf.reshape(patches_arr, [-1, IMG_X * IMG_Y * 3]).eval(),
    #y_: label_img_arr,
    keep_prob: 1.0})

result_dict = {}

for classIdx in validate_accuracy:
    if classIdx in c:
        key_grouped = c[classIdx]
        key_grouped = key_grouped[0:key_grouped.find('_')]
        if key_grouped in result_dict:
            result_dict[key_grouped]+=1
        else:
            result_dict[key_grouped]=1


result_sorted = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)
total_patches = len(validate_accuracy)

print("Filename:",fname.decode('UTF-8'), result_sorted, "Total: ", total_patches)

#Opens file for writing

with open(fname.decode('UTF-8')+".txt", 'w') as f:
  for name, count in result_sorted:
    print("{}: {:.2f}%".format(name, count*100/total_patches), file=f)


# In[14]:


#Plotting Stuffs

# with sess.as_default() as sess:
#     f = sess.run(tf.unstack(h_conv1, axis=-1), feed_dict = {
#         x: tf.reshape(patches_arr, [-1, IMG_X * IMG_Y * 3]).eval(),
#         keep_prob: 1.0})
#     
#     #print(f)
#     for i in range(30):
#         plt.figure(i, figsize=(2,2))
#         plt.imshow(f[30][i], interpolation='none', vmax=9)
#         plt.colorbar()
        
    #plt.show()


# 
