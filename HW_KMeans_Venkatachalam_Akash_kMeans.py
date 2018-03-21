"""
File name: kmeans.py
Language: Python3
Author :  Akash Venkatachalam

Description: The idea is to implement standard K-means algorithm and find the number of clusters present in the given
dataset. The given dataset has three attributes with all numeric values. We will extend the discussion by plotting
the Sum of Squared Errors (SSE) for all the clusters versus K, the number of clusters. The value of K will range
from 1 to 20, to see the variation in the plot and find the knee point value.

"""

import csv, math, sys, json
from random import randint
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def reading_in_csvfile():
    """
    Function to read in the CSV file and load in all the datapoints as a list
    :return: list of all the rows from the given dataset
    """
    with open("HW_2171_KMEANS_DATA__v502.csv") as csvFile:
        csvReader = csv.reader(csvFile)     # Reading in the file
        header = next(csvReader)            # Stroing the header
        dimension = len(header)             # Knowing the dimension of the dataset
        data_points_list = [row for row in csvReader]
        for item in data_points_list:       # Converting all the points from string to float
            item[0] = float(item[0])
            item[1] = float(item[1])
            item[2] = float(item[2])

        return data_points_list

def random_centroids(data_points_list, k):
    """
    Finding out the K (cluster size) initial random points from the set of given data points.
    :param data_points_list: given data points list
    :param k: the number of clusters that we need
    :return: assigning the initial random cluster centroids
    """
    initial_centroids_list = []
    try:
        for value in range(k):
            random_centroid = randint(0,len(data_points_list))     # Random centroid value from the dataset
            centroid_val = data_points_list[random_centroid]
            initial_centroids_list.append(centroid_val)            # appending the K chosen random points
        return  initial_centroids_list

    except IndexError:
        return [[7.99,3,6.56],[6.05,2.86,4.52],[2.27,5.82,4.56]]   # Randomly assigned a point in case of glitch

def euclidean_distance_three_dimension(one_list, second_list):
    """
    Function to find the euclidean distance between two 3-dimensional points
    :param one_list: List with three attributes
    :param second_list: List with three attributes
    :return: The euclidean distance value in float type
    """
    if (len(one_list) != len(second_list)):     # When the size of lengths are not same, ignore them
        return 100
    distance = math.sqrt( ((one_list[0] - second_list[0])**2) + ((one_list[1] - second_list[1])**2) + ((one_list[2] - second_list[2])**2) )
    return distance


def sse_evaluation(one_list, second_list):
    """
    Function to find the distance between the centroid and the datapoint of that cluster
    :param one_list: List with three attributes
    :param second_list: List with three attributes
    :return: The value of distance between them
    """
    if (len(one_list) and (len(second_list))) != 3:
        return 0
    dist = ( ((one_list[0] - second_list[0]))**2 + ((one_list[1] - second_list[1]))**2 + ((one_list[2] - second_list[2]))**2 )
    return dist



#######################################################################################################################

sse_list_to_plot = []
k_list = [k_val+1 for k_val in range(3)]  # Assigning the number of cluster value we want to observe like 1 to 20
for k in k_list:
    sse_list = []
    for best_sse in range(30):            # Finding the best SSE value by running the loop for 1000 times
        #k = 5
        data_points_list = reading_in_csvfile()                # Calling the function to read in the dataset
        centroids = random_centroids(data_points_list, k)      # Assigning random initial cluster values
        #print("\nInitial centoids:",centroids)
        new_centroid_list = centroids                          # Fixing the new cluster centroid values

        iter = 0             # number of iterations happening in a cluster
        while(iter != k+6):  # the stopping criteria for the movement of the centroid

            dict_for_clustering = {}
            cluster_value = 1
            for center in new_centroid_list:
                ip_for_dict_tostring = json.dumps(center)       # Using list for dict key and cluster number for value
                # json dumps is used for putting list as dictionary key
                dict_for_clustering[ip_for_dict_tostring] = cluster_value    # Initial assignment into dictionary
                cluster_value += 1

            for data_point in data_points_list:
                distance = sys.maxsize
                for centers in new_centroid_list:
                    new_distance = euclidean_distance_three_dimension(centers, data_point)
                    if (new_distance < distance):               # Finding the shortest distance with a point for clustering
                        distance = new_distance
                        dict_for_clustering[str(data_point)] = dict_for_clustering[str(centers)]

            changed_centorid_list = []
            for no_of_clusters in [c+1 for c in range(k)]:      # Iterating through the number of clusters
                for_np_list = []
                for point, cluster in dict_for_clustering.items():  # Iterating through the dictionary
                    if cluster == no_of_clusters:                   # Matching the clsuter and appending in order
                        for_np_list.append(json.loads(point))       # json loads is used for reading the dict key of list type

                new_val = np.array(for_np_list, ndmin = 3)          # Creating a 3d numpy array for each cluster to calculate the centroid
                new_cent = np.mean(new_val, axis= 1)
                changed_centorid_list.append(new_cent[0].tolist())  # New computed centorid based on the existing clustering
            new_centroid_list = changed_centorid_list
            iter += 1

        """
        After this we calculate the SSE for each K value. For this step, we need to find the SSE value at least 1000 times, since the
        initial centroid is chosen at random. We then fix the SSE of that K value by finding the mode of these 1000 obtained SSE values.
        """

        sse = 0
        no_of_clusterss = 1
        for centroid_val in new_centroid_list:
            sse_for_each_clust = 0
            num = 0
            for point, cluster in dict_for_clustering.items():
                if cluster == no_of_clusterss:  # SSE evaluation function is called for every individual cluster and summed up
                    sse += sse_evaluation(centroid_val, json.loads(point))  # json loads is used for reading the dict key of list type
                    num += 1
            no_of_clusterss += 1
            sse_for_each_clust = sse
            sse_for_each_clust = round(sse_for_each_clust, 1)
            print(k, new_centroid_list, num, round(sse_for_each_clust,1))   # Printing out the output

        sse_round = round(sse,2)        # SSE value rounded to two significant bits
        sse_list.append(sse_round)      # All 1000 SSE values appended to a list

    sse_numpy = np.array(sse_list)      # Converting this list to a numpy array
    mode = stats.mode(sse_numpy)        # Finding the most occurances of a SSE value
    #print("For",k, "clusters the SSE is:",float(mode[0]))
    sse_list_to_plot.append(float(mode[0]))  # Using this mode value to plot the graph SSE vs Cluster


print("\nThe SSE values for the clusters:", sse_list_to_plot)
plt.title('Plot of SSE vs Clusters')         # Setting up title for the graph
plt.xlabel('Clusters')                       # Displaying the x axis label
plt.ylabel('Sum of Squared Error (SSE)')     # Displaying the y axis label
plt.plot(k_list, sse_list_to_plot, 'ro-')    # Plotting the mixed variance vs threshold
plt.axis([0, 20, 0, 6000])
plt.grid(True)
plt.show()

#######################################################################################################################
