__author__ = 'Sagar Navgire'

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

# T - Number of minority class samples
# N% - Amount of SMOTE
# k - Number of nearest neighbors

# Sample[][] - Array for Original Minority class samlpes
# Synthetic[][] - Array for synthetic samples

numattrs = 36

def smoteforSatimage(T, N, k, minoritySamples, fo):
    if (N < 100):
        T = (N / 100) * T
        N = 100

    # Number of Synthetic samples to be created per sample from Minority class
    N = int(N / 100)

    for i in range(0, T):
        # print(" ")
        # print ("Minority Sample:" + str(minoritySamples[i]))
        printToFile(minoritySamples[i], fo)
        nnArray = getNearestNeighbors(minoritySamples, k + 1, i)
        Populate(N, i, k, nnArray, minoritySamples, fo)

    print ("Satimage Smoted - Synthetic samples added: " + str(N * T))


def Populate(N, currentIndex, k, nnArray, minoritySamples, fo):
    while (N != 0):
        nn = random.randint(1, k)
        # print ("NN Sample" + str(nnArray[nn]))
        for attr in range(1, numattrs):

            dif = 0
            Synthetic = []

            for i in range(0, 36):
                dif = float(nnArray[nn][i]) - float(minoritySamples[currentIndex][i])
                gap = random.uniform(0, 1)
                Synthetic.append(int(int(minoritySamples[currentIndex][i]) + (gap * dif)))

            Synthetic.append(1)

        # print ("Synthetic Sample:" + str(Synthetic))
        printToFile(Synthetic, fo)

        N = N - 1


def getNearestNeighbors(minoritySamples, k, i):
    X = np.array(minoritySamples)
    kNNModel = NearestNeighbors(k, 'auto').fit(X)
    distances, indices = kNNModel.kneighbors(np.array([minoritySamples[i]]))

    nnarray = []
    for index in indices[0]:
        nnarray.append(minoritySamples[index])

    return nnarray


def printToFile(Synthetic, fo):
    outputString = (
        str(Synthetic[0]) + "," + str(Synthetic[1]) + "," + str(Synthetic[2]) + "," + str(
            Synthetic[3]) + "," + str(Synthetic[4]) + "," + str(Synthetic[5]) + "," + str(
            Synthetic[6]) + "," + str(Synthetic[7]) + "," + str(Synthetic[8]) + "," + str(
            Synthetic[9]) + "," + str(Synthetic[10]) + "," + str(Synthetic[11]) + "," + str(
            Synthetic[12]) + "," + str(Synthetic[13]) + "," + str(Synthetic[14]) + "," + str(
            Synthetic[15]) + "," + str(Synthetic[16]) + "," + str(Synthetic[17]) + "," + str(
            Synthetic[18]) + "," + str(Synthetic[19]) + "," + str(Synthetic[20]) + "," + str(
            Synthetic[21]) + "," + str(Synthetic[22]) + "," + str(Synthetic[23]) + "," + str(
            Synthetic[24]) + "," + str(Synthetic[25]) + "," + str(Synthetic[26]) + "," + str(
            Synthetic[27]) + "," + str(Synthetic[28]) + "," + str(Synthetic[29]) + "," + str(
            Synthetic[30]) + "," + str(Synthetic[31]) + "," + str(Synthetic[32]) + "," + str(
            Synthetic[33]) + "," + str(Synthetic[34]) + "," + str(Synthetic[35])
        + "," + str(Synthetic[36])
    )
    fo.write(outputString + "\n")
