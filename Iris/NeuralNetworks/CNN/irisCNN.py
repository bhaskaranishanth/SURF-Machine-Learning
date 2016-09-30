import tensorflow as tf
import numpy as np
import sys
from functions import weight_variable
from functions import bias_variable
from functions import conv2d

# The size of each training batch for out Convolutional Neural Network 
# is a command line argument. 
BATCHSIZE = (int)(sys.argv[1])
TrainingSteps = 120/BATCHSIZE


# Start our interactive Tensor Flow session. 
sess = tf.InteractiveSession()

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, \
	target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, \
	target_dtype=np.int)

# Load training, test data into appropriate target.
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target

# Loading the data in a suitable fashion. 
trainingCorrect = np.genfromtxt(IRIS_TRAINING, delimiter = ',', skip_header =  1, \
	usecols = (4), dtype = np.int)
testCorrect = np.genfromtxt(IRIS_TEST, delimiter = ',', skip_header =  1, \
	usecols = (4), dtype = np.int)

# Now we will intialize one_hot vectors with the right classifications.
trainingY = np.zeros([trainingCorrect.size,3], np.int)
for i in range(trainingCorrect.size):
	trainingY[i][trainingCorrect[i]] += 1

testY = np.zeros([testCorrect.size,3], np.int)
for i in range(testCorrect.size):
	testY[i][testCorrect[i]] += 1



# Set up place holders for the test data as x and the correct labels
# stored in y_ which is a one_hot vector.
x = tf.placeholder(tf.float32, shape=[None, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

# FIRST CONVOLUTIONAL LAYER
# We are going to be modeling our IRIS data as a 2 x 2 x 1 image
# since we have 4 features. The first two dimensions are the patch
# size, the third is the number of input channels, and the last
# dimension is how many output channels we have. We also initialize
# a bias vector for each output channel.
W_conv1 = weight_variable([1, 1, 1, 32])
b_conv1 = bias_variable([32])

# We reshape x to a 4D tensor. The second and third dimensions refer
# to height, and the fourth dimension is number of color channels.
x_image = tf.reshape(x, [-1,2,2,1])

# Convolve the x_image with the weight tensor, apply the RELU function, 
# and then max-pool. Note that in this specific case we will not max-pool
# because max-pool is over 2x2 blocks and our entire "image" is a 2x2 block.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = h_conv1

# SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([1, 1, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = h_conv2

# DENSELY CONNECTED LAYER
# We add a fully-connected layer with 1024 neurons to allow processing
# on the entire image. We reshape the tensor from the pooling layer into 
# a batch of vectors, multiply by a weight matrix, add a bias, and apply a 
# ReLU.
W_fc1 = weight_variable([2 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DROPOUT
# To reduce overfitting, we will apply dropout before the readout layer. 
# We create a placeholder for the probability that a neuron's output is kept 
# during dropout. This allows us to turn dropout on during training, and turn 
# it off during testing. TensorFlow's tf.nn.dropout op automatically handles 
# scaling neuron outputs in addition to masking them, so dropout just works 
# without any additional scaling.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER
# Finally, we add a softmax layer,
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# TRAIN AND EVALUATE MODEL

# TRAINING MODEL
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),\
 reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(TrainingSteps):
  train_accuracy = accuracy.eval(feed_dict={x: training_set.data[BATCHSIZE * \
    i: BATCHSIZE * (i + 1)], y_: trainingY[BATCHSIZE * i: BATCHSIZE * \
    (i + 1)], keep_prob: 1.0})
  print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: training_set.data[BATCHSIZE * i: BATCHSIZE * \
    (i + 1)], y_: trainingY[BATCHSIZE * i: BATCHSIZE * (i + 1)], \
     keep_prob: 0.5})

# TESTING MODEL ON TEST SET
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: x_test, y_: testY, keep_prob: 1.0}))

# Printing out the probabilities of the classification for each one.
probMatrix = y_conv.eval(feed_dict={x: x_test, y_:testY, keep_prob: 1.0})
print(probMatrix)


#for i in range(len(probMatrix)):
#  print(str(probMatrix[i][0]) + ',' + str(probMatrix[i][1]) + ','  \
#    + str(probMatrix[i][2]) + ',' + str(testCorrect[i]))
confusionMatrix = np.zeros((2,2), dtype = np.int)
for i in range(len(probMatrix)):
  maxIndex = 0
  for j in range(3):
    if ((probMatrix[i][j]) > (probMatrix[i][maxIndex])):
      maxIndex = j
  confusionMatrix[maxIndex][testCorrect[i]] += 1
print(testCorrect)
print(confusionMatrix)


