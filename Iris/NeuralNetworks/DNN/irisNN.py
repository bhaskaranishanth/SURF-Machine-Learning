import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, \
	target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, \
	target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
print(x_train.shape)
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10],\
 n_classes=3)

# Fit model.
classifier.fit(x=x_train, y=y_train, steps=200)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


# Compute Confusion Matrix
testData = np.genfromtxt(IRIS_TEST, delimiter = ',', skip_header = 1, \
	usecols = (0,1,2,3))
testCorrect = np.genfromtxt(IRIS_TEST, delimiter = ',', skip_header =  1, \
	usecols = (4), dtype = np.int)
pred = classifier.predict(testData)
confusionMatrix = np.zeros((3,3), dtype = np.int)
for i in range(pred.size):
	confusionMatrix[testCorrect[i],pred[i]] += 1
print(confusionMatrix)
#GRAPHICAL REPRESENTATION BELOW. OUTPUT TO TERMINAL ABOVE
#columns = ('Correct C1', 'Correct C2', 'Correct C3')
#rows = ('Predicted C1', 'Predicted C2', 'Predicted C3')
#table = plt.table(cellText = confusionMatrix, rowLabels = rows, \
#	colLabels = columns, loc = 'center')
#plt.subplots_adjust(left = 0.2, top = 0.8)
#plt.show()

# Show the probabilities of the classification 
probabilities = classifier.predict_proba(testData)
for i in range(len(probabilities)):
	print(str(probabilities[i][0]) + ',' + str(probabilities[i][1]) + ','  \
		+ str(probabilities[i][2]) + ',' + str(testCorrect[i]))

