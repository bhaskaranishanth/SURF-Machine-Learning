import tensorflow as tf 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import csv 
import pandas
import seaborn
from numpy import array
from functions import read_datafile
from functions import TensorFlowKMeans
from functions import majority
from functions import assignProperLabels
import copy

# First of all, we need to make the important note that in the .csv file, we 
# have changed the class labels from their name to numbers in order for ease 
# when dealing with our data set in our code. We use the following mapping:
# Iris Setosa --> 0, Iris Versicolour --> 1, Iris Virginica --> 2


titles =['S_Length', 'S_Width', 'P_Length', 'P_Width', 'Class']

try: 
	data = read_datafile('iris.data.csv', titles)
except ValueError:
	print('Too many labels for too few columns :(')
	exit()

# data2 will be used for our kMeans calculations

data2 = np.genfromtxt('iris.data.csv', delimiter=',',usecols = (0,1,2,3))
copyData = copy.deepcopy(data)

# We run the kMeans algorithm here and save our results into assignments
centroids, assignments = TensorFlowKMeans(data2, 3)

# Here we format our results correctly, and have copyData have our new results.
kMeansResults = assignProperLabels(assignments, 50, 3, 3)
copyData['Class'] = kMeansResults


# Since there are 4 characteristics and we can only depict results on a 2-D plot, 
# we will need 6 graphs to accomadate every single combination of parameters. 


# First let's take care of the charactersitics that will be the same for all of 
# our plots. This will be our legend.
white_patch = mpatches.Patch(color = 'white', label='Iris Setosa')
gray_patch = mpatches.Patch(color = 'gray', label = 'Iris Versicolour')
black_patch = mpatches.Patch(color = 'black', label = 'Iris Virginica')

'''
# GRAPH GROUP 1

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Sepum Length Versus Sepum Width")
plt.xlabel("Sepum Length")
plt.ylabel("Sepum Width")
plt.scatter(data['S_Length'], data['S_Width'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Sepum Length Versus Sepum Width")
plt.xlabel("Sepum Length")
plt.ylabel("Sepum Width")
plt.scatter(data['S_Length'], data['S_Width'], c = copyData['Class'], s=50)
plt.gray()
plt.show()

# GRAPH GROUP 2

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Sepum Length Versus Petal Length")
plt.xlabel("Sepum Length")
plt.ylabel("Petal Length")
plt.scatter(data['S_Length'], data['P_Length'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Sepum Length Versus Petal Length")
plt.xlabel("Sepum Length")
plt.ylabel("Petal Length")
plt.scatter(data['S_Length'], data['P_Length'], c = copyData['Class'], s=50)
plt.gray()
plt.show()

# GRAPH GROUP 3 
plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Sepum Length Versus Petal Width")
plt.xlabel("Sepum Length")
plt.ylabel("Petal Width")
plt.scatter(data['S_Length'], data['P_Width'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Sepum Length Versus Petal Width")
plt.xlabel("Sepum Length")
plt.ylabel("Petal Width")
plt.scatter(data['S_Length'], data['P_Width'], c = copyData['Class'], s=50)
plt.gray()
plt.show()

# GRAPH GROUP 4
plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Sepum Width Versus Petal Length")
plt.xlabel("Sepum Width")
plt.ylabel("Petal Length")
plt.scatter(data['S_Width'], data['P_Length'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Sepum Width Versus Petal Length")
plt.xlabel("Sepum Width")
plt.ylabel("Petal Length")
plt.scatter(data['S_Width'], data['P_Length'], c = copyData['Class'], s=50)
plt.gray()
plt.show()
'''

#GRAPH GROUP 5
plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Sepum Width Versus Petal Width")
plt.xlabel("Sepum Width")
plt.ylabel("Petal Width")
plt.scatter(data['S_Width'], data['P_Width'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Sepum Width Versus Petal Width")
plt.xlabel("Sepum Width")
plt.ylabel("Petal Width")
plt.scatter(data['S_Width'], data['P_Width'], c = copyData['Class'], s=50)
plt.gray()
plt.show()

'''
#GRAPH GROUP 6
plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("Actual Petal Length Versus Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter(data['P_Length'], data['P_Width'], c = data['Class'], s=50)
plt.gray()
plt.show()

plt.legend(handles=[white_patch, gray_patch, black_patch])
plt.title("KMeans Petal Length Versus Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter(data['P_Length'], data['P_Width'], c = copyData['Class'], s=50)
plt.gray()
plt.show()
'''

# Now we will output the accuracy of our kMeans results
print assignments
print kMeansResults
wrongTally = 0
for i in range(50):
	if (kMeansResults[i] != int((i / 50))):
		wrongTally += 1
accuracy = 1 - (float(wrongTally)/50)
print('The accuracy for kMeans for the first class is ' + str(accuracy))

wrongTally = 0
for i in range(50,100):
	if (kMeansResults[i] != int((i / 50))):
		wrongTally += 1
accuracy = 1 - (float(wrongTally)/50)
print('The accuracy for kMeans for the second class is ' + str(accuracy))

wrongTally = 0
for i in range(50,100):
	if (kMeansResults[i] != int((i / 50))):
		wrongTally += 1
accuracy = 1 - (float(wrongTally)/50)
print('The accuracy for kMeans for the third class is ' + str(accuracy))


