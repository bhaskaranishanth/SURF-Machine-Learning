import tensorflow as tf 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import csv 
import pandas
import seaborn
from random import choice, shuffle
from numpy import array
import copy


'''
K-Means Clustering using TensorFlow.
'vectors' should be a n*k 2-D NumPy array, where n is the number
of vectors of dimensionality k.
'kClusters' should be an integer.
'''
def TensorFlowKMeans(vectors, kClusters):
 
    kClusters = int(kClusters)
    assert kClusters < len(vectors)
 
    # First we find the dimensionality of our data set.  
    dim = len(vectors[0])
 
    # Let's just shuffle the indices to select random centroids. 

    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)
 
    # We initialize a new graph and set it as the default during each run
    # of this algorithm. This will allow us not to crowd our graph with 
    # unused/unnecessary operations or values.
 
    graph = tf.Graph()
 
    with graph.as_default():
 
        sess = tf.Session()
 
        # First lets ensure we have a Variable vector for each centroid,
        # initialized to one of the vectors from the available data points

        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(kClusters)]

        # These nodes will assign the centroid Variables the appropriate
        # values

        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
 
        # Variables for cluster assignments of individual vectors(initialized
        # to 0 at first)

        assignments = [tf.Variable(0) for i in range(len(vectors))]

        # These nodes will assign an assignment Variable the appropriate
        # value.

        assignment_value = tf.placeholder("int32")
        cluster_assigns = []

        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))
 
        # Now lets construct the node that will compute the mean
        # The placeholder for the input

        mean_input = tf.placeholder("float", [None, dim])

        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors

        mean_op = tf.reduce_mean(mean_input, 0)
 
        # Node for computing Euclidean distances
        # Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            v1, v2), 2)))
 
        # This node will figure out which cluster to assign a vector to,
        # based on Euclidean distances of the vector from the centroids.
        # Placeholder for input

        centroid_distances = tf.placeholder("float", [kClusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

 
        # This will help initialization of all Variables defined with respect
        # to the graph. The Variable-initializer should be defined after
        # all the Variables have been constructed, so that each of them
        # will be included in the initialization.

        init_op = tf.initialize_all_variables()
 
        # Initialize all variables
        sess.run(init_op)
 
        # Now perform the Expectation-Maximization steps of K-Means clustering
        # iterations. To keep things simple, we will only do a set number of
        # iterations, instead of using a Stopping Criterion.
        kIterations = 100
        for iteration_n in range(kIterations):

            # Based on the centroid locations till last iteration, compute
            # the _expected_ centroid assignments.
            # Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # Compute Euclidean distance between this vector and each
                # centroid. Remember that this list cannot be named
                # 'centroid_distances', since that is the input to the
                # cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                # Now use the cluster assignment node, with the distances
                # as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                # Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
 
            # Based on the expected state computed from the Expectation Step,
            # compute the locations of the centroids so as to maximize the
            # overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(kClusters):
                # Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                # Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                # Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})
 
        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments


''' 
This function takes in two arguments. The first argument is the name of the 
.csv file from which it will read. The second argument is an array of strings
which correspond to the titles of each column it is reading from. The length
of this array should obviously be less than the number of columns in 
.csv file from which we are reading from.
'''
def read_datafile(file_name, titles):
	# First we make sure we don't have too many labels for too few columns.
	f = open(file_name, 'r')
	reader = csv.reader(f, delimiter = ',')
	nCol = len(next(reader))
	nActualCol = len(titles)
	if nCol < nActualCol:
		raise ValueError()
	f.close()
	# Now that we handled that potential error, we read in the data and assign
	# appropriate labels.  
	data = np.genfromtxt(file_name, delimiter=',', names = titles)
	return data

'''
Given an array of assignments, this will reassign the labels as numbers between
0 to k - 1, with cluster 1 corresponding to label 0, and cluster k corresponding
to label k - 1.
'''
def assignProperLabels(assignments, numberPerSample, kClusters, maxLabels):
    indexes = []
    toChange = []
    changedIndexes = []
    currentIndex = 0
    clone = copy.deepcopy(assignments)
    while True:
        if (currentIndex == len(assignments)):
            break
        else:
            indexes.append(currentIndex)
            currentIndex += ((len(assignments)) / kClusters)
    for i in range(len(indexes)):
        number = majority(assignments, indexes[i], 
                indexes[i] + (len(assignments) / numberPerSample), 
                maxLabels)
        toChange.append(number)
    for j in range(len(toChange)):
        for k in range(len(assignments)):
            if k in changedIndexes:
                pass
            else:
                if clone[k] == toChange[j]:
                    clone[k] = j
                    changedIndexes.append(k)
    return clone





            

'''
Helper function for assignProperLabels()
'''
def majority(vector, startingIndex, endingIndex, maxLabels):
    tallyVector = []
    maxLabel = 0
    maxTally = 0
    for i in range(maxLabels):
        tallyVector.append(0)
    for index in range(startingIndex,endingIndex):
        tallyVector[vector[index]] = tallyVector[vector[index]] + 1
    for i in range(len(tallyVector)):
        if tallyVector[i] > tallyVector[maxTally]:
            maxLabel = tallyVector[i]
            maxTally = i
    return maxTally 



