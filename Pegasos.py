import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv
from random import randint


def Pegasos(X, Y, T, lbda=.5):
    """Performs the Pegasos algorithm for T iterations and returns the weights.
    There is no specfication for the dimensionality. It should be noted that this 
    implementation follows the one outlined in the notes, and not some others.

    Args:
        X (ndarray): N dimensional array containing the data
        Y (ndarray): N dimensional array of same length and column size 1. Corresponds to group of items.
        T (int): Number of iterations
        lbda (float, optional): lambda parameter for Pegasos algorithm. Defaults to .5. Lower lambdas arrive at harder solutions

    Returns:
        Ndarray: weights for each element along an axis
    """
    XBar = np.append(X, np.ones([len(X), 1]), 1)
    WBar = np.zeros(XBar[0].shape)
    W = np.zeros(XBar[0].shape)
    S = len(Y)
    Theta = np.zeros(XBar[0].shape)
    for iter in range(1, T):
        if iter != 1:
            W = 1/(lbda*(iter-1))*Theta
        item = randint(0, S-1)
        if Y[item]*np.dot(W, XBar[item].T) < 1:
            Theta = Theta + Y[item]*XBar[item]

        WBar = (1-1/iter)*WBar + (1/iter)*W
        # There is no need to implement the else case

    return WBar


if __name__ == "__main__":
    # defining the paths for training and testing data
    testDataPath = "./testData.csv"
    trainDataPath = "./testData.csv"

    # Training Data and Testing Data
    trainData = []
    testData = []

    # setting up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # constants for the pegasos algorithm
    T = 100000
    lbda = .0001

    # populating the arrays
    with open(trainDataPath, 'r') as trainFile:
        csvreader = csv.reader(trainFile)
        for idx, row in enumerate(csvreader):
            trainData.append(row)

    with open(testDataPath, 'r') as testFile:
        csvreader = csv.reader(testFile)
        for idx, row in enumerate(csvreader):
            testData.append(row)

    # plotting the points
    trainData = np.asarray(trainData, float)
    testData = np.asarray(testData, float)

    group1Train = (trainData[:, -1] == 1)
    group1Test = (testData[:, -1] == 1)

    group2Train = (trainData[:, -1] == -1)
    group2Test = (testData[:, -1] == -1)

    ax.scatter(trainData[group1Train][:, 0], trainData[group1Train]
               [:, 1], trainData[group1Train][:, 2], color="red")
    ax.scatter(trainData[group1Test][:, 0], trainData[group1Test]
               [:, 1], trainData[group1Test][:, 2], color='red', marker='x')
    ax.scatter(testData[group2Train][:, 0], testData[group2Train]
               [:, 1], testData[group2Train][:, 2], color="green")
    ax.scatter(testData[group2Test][:, 0], testData[group2Test]
               [:, 1], testData[group2Test][:, 2], color="green", marker='x')

    # Caclulating the Hyperplane using Pegasos
    weights = Pegasos(testData[:, :3], testData[:, -1], T, lbda)

    # plotting the plane
    Xp = np.linspace(-1.1*np.min(testData[:, 0]),
                     1.1*np.max(testData[:, 0]), 10)
    Yp = np.linspace(-1.1*np.min(testData[:, 1]),
                     1.1*np.max(testData[:, 1]), 10)
    Xp, Yp = np.meshgrid(Xp, Yp)
    Zp = (weights[0]*Xp+weights[1]*Yp + weights[3])/(-1*weights[2])
    ax.plot_surface(Xp, Yp, Zp, alpha=.5, color="purple")

    totalPts = len(testData)
    incorrect = 0
    for pt in testData:
        estimate = weights[0]*pt[0] + weights[1] * \
            pt[1] + weights[2]*pt[2] + weights[3]
        if estimate > 0:
            estimate = 1
        else:
            estimate = -1

        if estimate != pt[-1]:
            incorrect += 1

    plt.title("Lambda: "+str(lbda)+" Error: "+str(incorrect/totalPts))
    plt.show()
