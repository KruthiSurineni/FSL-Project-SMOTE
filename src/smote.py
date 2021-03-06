
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

# T - Number of minority class samples
# N% - Amount of SMOTE
# k - Number of nearest neighbors

# Sample[][] - Array for Original Minority class samples
# Synthetic[][] - Array for synthetic samples

newIndex = 1
numattrs = 7


def smote(T, N, k, minoritySamples, fo):
    print ("In smote")

    if (N < 100):
        T = (N / 100) * T
        N = 100

    # Number of Synthetic samples to be created per sample from Minority class
    N = int(N / 100)
    print("Vaue of N: " + str(N))

    for i in range(0, T - 1):
        # print(" ")
        printToFile(minoritySamples[i], fo)
        nnArray = getNearestNeighbors(minoritySamples, k + 1, i)
        Populate(N, i, k, nnArray, minoritySamples, fo)

        # fo.close()


def Populate(N, currentIndex, k, nnArray, minoritySamples, fo):
    while (N != 0):
        nn = random.randint(1, k)
        # print ("NN Sample" + str(nnArray[nn]))
        for attr in range(1, numattrs):

            dif = 0
            Synthetic = []

            for i in range(0, 8):
                gap = random.uniform(0, 1)
                if (i == 5 or i == 6):
                    dif = float(nnArray[nn][i]) - float(minoritySamples[currentIndex][i])
                    if (i == 5):
                        Synthetic.append(round(float(minoritySamples[currentIndex][i]) + (gap * dif), 2))
                    else:
                        Synthetic.append(round(float(minoritySamples[currentIndex][i]) + (gap * dif), 3))
                else:
                    dif = int(nnArray[nn][i]) - int(minoritySamples[currentIndex][i])
                    Synthetic.append(int(int(minoritySamples[currentIndex][i]) + (gap * dif)))

            Synthetic.append(1)

        # print ("Minority Sample:" + str(minoritySamples[i]))
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
    outputString = ''.join(str(s) for s in Synthetic)
    fo.write(outputString + "\n")
