
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

# T - Number of minority class samples
# N% - Amount of SMOTE
# k - Number of nearest neighbors

# Sample[][] - Array for Original Minority class samlpes
# Synthetic[][] - Array for synthetic samples

numattrs = 5


def smoteforPhoneme(T, N, k, minoritySamples, fo):
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

    print ("Phoneme Smoted - Synthetic samples added: " + str(N * T))


def Populate(N, currentIndex, k, nnArray, minoritySamples, fo):
    while (N != 0):
        nn = random.randint(1, k)
        # print ("NN Sample" + str(nnArray[nn]))
        for attr in range(1, numattrs):

            dif = 0
            Synthetic = []

            for i in range(0, 5):
                dif = float(nnArray[nn][i]) - float(minoritySamples[currentIndex][i])
                gap = random.uniform(0, 1)
                Synthetic.append(round(float(minoritySamples[currentIndex][i]) + (gap * dif), 6))

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
            Synthetic[3]) + "," + str(Synthetic[4]) + "," + str(Synthetic[5]))
    fo.write(outputString + "\n")
