import numpy as np
import findspark
from pyspark import SparkContext, SparkConf
from operator import add
import math
import time
import matplotlib.pyplot as plt
import operator

findspark.init("/home/elhammirafzali/spark-2.4.0-bin-hadoop2.7/")
conf = SparkConf() \
    .setAppName("PySpark App") \
    .setMaster("local[*]") \
    .set("spark.executor.heartbeatInterval", "1h") \
    .set("spark.network.timeout", "10000001s") \
    .set("spark.executor.memory", "8g")
sc = SparkContext(conf=conf)
sc.setLogLevel("ALL")


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(1, length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x + 1][0][0])
    return neighbors


def cal_k_neighbour(x, matrix, k):
    neighbour = get_neighbors(matrix, x, k)
    return neighbour


def new_map_reduce(matrix, k, map_num):
    n = len(matrix)
    keys = np.array([range(n)]).reshape([n, 1])
    matrix = np.concatenate((keys, matrix), axis=1)

    start = time.time()
    rdd = sc.parallelize(matrix, map_num)
    rdd2 = rdd.map(lambda x: cal_k_neighbour(x, matrix, k)).collect()
    exe_time = time.time() - start
    np.save('neighbours/neighbours.npy', rdd2)
    neighbours = np.reshape(rdd2, (n, k))

    return neighbours.astype(int), exe_time


def calculate_distance(x):
    a = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a.append(((i, j), (x[i] - x[j]) ** 2))
            a.append(((j, i), (x[i] - x[j]) ** 2))
    return a


def reformat_rdd(x):
    return x[0][0], (x[0][1], x[1])


def find_k_nearest_neighbours(matrix, k, map_num):
    start = time.time()
    n = len(matrix[0])
    # first map-reduce
    rdd = sc.parallelize(matrix, map_num)
    distance_rdd = rdd.flatMap(lambda x: calculate_distance(x))
    sum_rdd = distance_rdd.reduceByKey(add).mapValues(lambda x: math.sqrt(x))

    # second map-reduce
    reformatted_rdd = sum_rdd.map(lambda x: (reformat_rdd(x)))
    result_rdd = reformatted_rdd.sortBy(lambda y: y[1][1]).groupByKey().map(
        lambda x: (x[0], list(x[1]))).sortByKey(ascending=True).collect()

    # fill the output matrix
    neighbours = np.array([[]])

    for i in range(n):
        dist_array = result_rdd[i][1]
        temp = []
        for j in range(k):
            temp.append(dist_array[j][0])
        neighbours = np.append(neighbours, temp)

    neighbours = np.reshape(neighbours, (n, k))

    exe_time = time.time() - start
    return neighbours.astype(int), exe_time


def calculate_speed_up():
    c = []
    t = []
    matrix = np.load("features_reduced/features_reduced.npy")
    matrix = matrix[:3000]
    k = 5
    for i in range(2, 12):
        partition_num = 4
        core_number = i
        conf.setMaster("local[" + str(core_number) + "]")
        _, execute_time = new_map_reduce(matrix, k, partition_num)
        c.append(core_number)
        t.append(execute_time)

    plt.xlabel('map(thread) number')
    plt.ylabel('execution time')
    plt.plot(c, t)
    plt.grid(axis='x', linestyle='-', color='#777777')
    plt.grid(axis='y', linestyle='-', color='#777777')
    plt.show()

# calculate_speed_up()
