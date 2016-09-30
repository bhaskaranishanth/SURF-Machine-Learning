import os
import numpy as np 
import nibabel as nib 
import sys
from time import sleep
import tensorflow as tf 
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

# Here we start an interactive TensorFlow session.
sess = tf.InteractiveSession()

# This function will reformat our data so that we can take better advantage
# of separating our time intervals.
def reformatData(originalClip):
  newClip = np.zeros((300,64,64,40))
  print
  print("Reformatting Image Data")
  for i in range(64):
    printProgress(i, 63, prefix = 'Progress:', suffix = 'Complete', 
    barLength = 50)
    for j in range(64):
      for k in range(40):
        for l in range(300):
          newClip[l][i][j][k] = originalClip[i][j][k][l]
  return newClip


# This function will get all of the immediate subdirectories and load them
# as strings into an array.

def get_immediate_subdirectories(a_dir):
  return [name for name in os.listdir(a_dir)
    if os.path.isdir(os.path.join(a_dir, name))]

# This function will help us visualize progress because a lot of this data will 
# take a long time to load and process. 

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 2, 
  barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '0' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', 
      suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# This function will help us get specific batches from all file. The "batchnumber" is a
# number between 0 and 23 (inclusive) if we are looking at training data, and is a -1
# if we are retrieving testData. The second, third, and fourth arguments are template
# strings which indicate where the NIFTI files are. The fifth argument, "allFolder", is
# an array containing strings which are the titles of all of the data we need. The fifth
# argument, "BatchesIndices" is an ordering of the indices of the folders we need, and
# "testBatchIndices" is simply this ordering of indices except for the test batch.

def getBatch(batchNumber, templateStr1, templateStr2, templateStr3, 
  allFolder, BatchesIndices, testBatchIndices):
  if (batchNumber >= 0):
    FullData = np.zeros((5,300,64,64,40))
    #FullData = np.zeros((5,64,64,40,300))
    print('Importing MRI Images Batch ' + str(batchNumber))
    counter = 0
    printProgress(counter, 5, prefix = 'Progress:', suffix = 'Complete', 
      barLength = 50)
    for i in range(5):
      folder = allFolder[BatchesIndices[batchNumber,i]]
      filename = templateStr1 + folder + templateStr2 + folder + templateStr3
      img = nib.load(filename)
      FullData[counter] = reformatData(img.get_data())
      #FullData[counter] = img.get_data()
      sleep(0.1)
      counter += 1
      printProgress(counter, 5, prefix = 'Progress:', suffix = 'Complete', 
        barLength = 50)
    print
  else:
    print('Importing Test Batch')
    #FullData = np.zeros((25,64,64,40,300))
    FullData = np.zeros((25,300,64,64,40))
    counter = 0
    printProgress(counter, 25, prefix = 'Progress:', suffix = 'Complete', 
      barLength = 50)
    for i in range(len(testBatchIndices)):
      folder = allFolder[testBatchIndices[i]]
      filename = templateStr1 + folder + templateStr2 + folder + templateStr3
      img = nib.load(filename)
      aFullData[counter] = reformatData(img.get_data())
      #FullData[counter] = img.get_data()
      sleep(0.1)
      counter += 1
      printProgress(counter, 25, prefix = 'Progress:', suffix = 'Complete', 
        barLength = 50)
    print
  return FullData

# First we get all of the folders of all the patient data and load into 
# allFolder.

allFolder = get_immediate_subdirectories('./ABIDE_UM')

# The data is too big to be loaded all at once, so we will load it in batches
# of 5 pictures at a time.

#CurrentBatch = np.zeros((5,64,64,40,300))
CurrentBatch = np.zeros((5,300,64,64,40))

# The template strings accompanied by the names in allFolder will be from where
# we are pulling the MRI data. 

templateStr1 = './ABIDE_UM/'
templateStr2 = '/'
templateStr3 = '/scans/rest/resources/NIfTI/files/rest.nii'

# Let's load the labels before we load the data.
file1 = './ABIDE_UM/phenotypic_UM_1.csv'
file2 = './ABIDE_UM/phenotypic_UM_2.csv'
FirstLabels = np.genfromtxt(file1, delimiter = ',', usecols = (2), 
  skip_header = 1, dtype = np.int)
SecondLabels = np.genfromtxt(file2, delimiter = ',', usecols = (2), 
  skip_header = 1, dtype = np.int)
Labels = np.concatenate([FirstLabels, SecondLabels], axis = 0)

# Now we have to make the labels in the shape of a one-hot vector.
AllLabels = np.zeros((Labels.size, 2), np.int)
for i in range(Labels.size):
  AllLabels[i][Labels[i] - 1] += 1

# Before we do anything, we will set up our training so that each
# batch has samples of both 1's and 0's.
zeroIndices = []
oneIndices = []
for i in range(len(AllLabels)):
  if (Labels[i] == 1):
    zeroIndices.append(i)
  else:
    oneIndices.append(i)

# We now save the indices in batches. We will have 24 batches each
# with 5 pictures each, and then a test batch with 25 pictures. 

BatchesIndices = np.zeros((24,5),np.int)
currentZeroIndex = 0
currentOneIndex = 0
for i in range(24):
  if (i < 18):
    BatchesIndices[i][0] = (zeroIndices[currentZeroIndex])
    BatchesIndices[i][1] = (zeroIndices[currentZeroIndex + 1])

    BatchesIndices[i][2] = (oneIndices[currentOneIndex])
    BatchesIndices[i][3] = (oneIndices[currentOneIndex + 1])
    BatchesIndices[i][4] = (oneIndices[currentOneIndex + 2])

    currentZeroIndex += 2
    currentOneIndex += 3
  else:
    BatchesIndices[i][0] = (zeroIndices[currentZeroIndex])
    BatchesIndices[i][1] = (zeroIndices[currentZeroIndex + 1])
    BatchesIndices[i][2] = (zeroIndices[currentZeroIndex + 2])

    BatchesIndices[i][3] = (oneIndices[currentOneIndex])
    BatchesIndices[i][4] = (oneIndices[currentOneIndex + 1])

    currentZeroIndex += 3
    currentOneIndex += 2

# Now we get the proper labels for the batches that we are using.

LabelsIndices = np.zeros((24,5),np.int)
for i in range(24):
  for j in range(5):
    LabelsIndices[i,j] = Labels[BatchesIndices[i,j]] - 1

# Now we set up the test batch.

testBatchIndices = []
while (currentZeroIndex < len(zeroIndices)):
  testBatchIndices.append(zeroIndices[currentZeroIndex])
  currentZeroIndex += 1
while (currentOneIndex < len(oneIndices)):
  testBatchIndices.append(oneIndices[currentOneIndex])
  currentOneIndex += 1

# Now we set up the labels for the testBatch.

testLabelsIndices = []
for i in range(len(testBatchIndices)):
  testLabelsIndices.append(Labels[testBatchIndices[i]] - 1)

# Lastly, we set them up as one-hot vectors. 
testL = np.zeros((25,2), np.int)
for i in range(25):
  testL[i][testLabelsIndices[i]] += 1

trainingL = np.zeros((24,5,2), np.int)
for i in range(24):
  for j in range(5):
    trainingL[i][j][LabelsIndices[i,j]] += 1

# If you want to visualize any of the layers of the MRI data, you can view it
# here. 

'''
img = np.zeros((64,64), np.float32)

for i in range(64):
  for j in range(64):
    img[i][j] = CurrentBatch[0][i][j][10][150]
print(img)

height, width = 64, 64 #in pixels
spines = 'left', 'right', 'top', 'bottom'

labels = ['label' + spine for spine in spines]

tick_params = {spine : False for spine in spines}
tick_params.update({label : False for label in labels})

img = img

desired_width = 8 #in inches
scale = desired_width / float(width)

fig, ax = plt.subplots(1, 1, figsize=(desired_width, height*scale))
img = ax.imshow(img, cmap=cm.Greys_r, interpolation='none')

#remove spines
for spine in spines:
    ax.spines[spine].set_visible(False)

#hide ticks and labels
ax.tick_params(**tick_params)

#preview
plt.show()

#save
fig.savefig('test.png', dpi=300)

'''

# Now we finally start setting up our convolutional neural network. 
# x = tf.placeholder(tf.float32, shape=[None, 64, 64, 40, 300])
x = tf.placeholder(tf.float32, shape=[None, 300, 64, 64, 40])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


# To create this model, we're going to need to create a lot of weights and biases.
# One should generally initialize weights with a small amount of noise for symmetry 
# breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also 
# good practice to initialize them with a slightly positive initial bias to avoid 
# "dead neurons." Instead of doing this repeatedly while we build the model, let's 
# create two handy functions to do it for us. 

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Our pooling is plain old max pooling over 2x2 blocks. 
# To keep our code cleaner, let's also abstract this operation into a function.


def max_pool_2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                        strides=[1, 1, 2, 2, 1], padding='SAME')

# STRUCTURE OF NETWORK

# INPUT VOLUME = (300, 64, 64, 40)
# CONV LAYER = (10, 32, 32, 40, 64)
#   output of this will be (30, 32, 32, 64)
# POOL LAYER
#   output of this will be (30, 16, 16, 64)
# RELU LAYER. 
# CONV LAYER = (10, 8, 8, 64, 128)
#   output of this will be (3, 8, 8, 128)
# POOL LAYER 
#   output of this layer will be (3, 4, 4, 128)
# REUL LAYER
# CONV LAYER = (3, 2, 2, 128, 256)
#   output of this layer will be (1, 2, 2, 256)
# FC LAYER 4096 Neurons.
# BINARY OUTPUT

# Implementation of the above can be seen below.

# Here we start an interactive TensorFlow session.
sess = tf.InteractiveSession()

# FULL NETWORK

W_conv1 = weight_variable([10,8,8,40,64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x,[-1,300,64,64,40])
h_conv1 = tf.nn.conv3d(x_image, W_conv1, strides=[1, 10, 2, 2, 1], 
  padding='SAME')
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([10,8,8,64,128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.conv3d(h_pool1, W_conv2, strides=[1, 10, 2, 2, 1],
  padding='SAME')

h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 2, 2, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 =tf.nn.conv3d(h_pool2, W_conv3, strides=[1, 3, 2, 2, 1],
  padding='SAME')
W_fc1 = weight_variable([2 * 2 * 256, 4096])
b_fc1 = bias_variable([4096])

h_pool3_flat = tf.reshape(h_conv3, [-1, 2 * 2 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([4096, 2])
b_fc2 = bias_variable([2])



# THIS IS THE SHORTER VERSION OF THE NEURAL NETWORK. IT WILL NOT BE AS
# EFFICIENT AS WANT IT TO BE. 
'''
W_conv1 = weight_variable([4,4,40,300,8])
b_conv1 = bias_variable([8])
x_image = tf.reshape(x, [-1,64,64,40,300])
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([4, 4, 1, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([4 * 4 * 16, 32])
b_fc1 = bias_variable([32])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([32, 2])
b_fc2 = bias_variable([2])


'''
# THIS VERSION IS WHEN WE REFORMAT THE DATA . WILL TAKE LONGER. 
'''
W_conv1 = weight_variable([100, 4, 4, 40, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1,300,64,64,40])
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 4, 4, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 4 * 32, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4* 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 2])
b_fc2 = bias_variable([2])
'''

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), 
  reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(24):
  logging.getLogger().setLevel(logging.INFO)
  CurrentBatch = getBatch(i, templateStr1, templateStr2, templateStr3, allFolder, 
    BatchesIndices, testBatchIndices)

  train_accuracy = accuracy.eval(feed_dict={x: CurrentBatch, y_: 
    trainingL[i], keep_prob: 1.0})

  print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: CurrentBatch, y_: trainingL[i]
   , keep_prob: 0.5})

CurrentBatch = np.zeros([25,300,64,64,40])
CurrentBatch = getBatch(-1, templateStr1, templateStr2, templateStr3, allFolder, 
    BatchesIndices, testBatchIndices)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: CurrentBatch, y_: testL, keep_prob: 1.0}))

CurrentBatch = np.zeros((25,300,64,64,40))
CurrentBatch = getBatch(-1, templateStr1, templateStr2, templateStr3, allFolder,
  BatchesIndices, testBatchIndices)

# Printing out the probabilities of the classification for each one.
probMatrix = y_conv.eval(feed_dict={x: CurrentBatch, y_: testL, keep_prob: 1.0})
print(probMatrix)


# Now we need to print out the confusion matrix to see exactly where
# our errors lie.
confusionMatrix = np.zeros((2,2), dtype = np.int)
for i in range(len(probMatrix)):
  maxIndex = 0
  for j in range(2):
    if ((probMatrix[i][j]) > (probMatrix[i][maxIndex])):
      maxIndex = j
  confusionMatrix[maxIndex][LabelsIndices[i]] += 1
print(confusionMatrix)
