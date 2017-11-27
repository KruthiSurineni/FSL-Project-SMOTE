__author__ = 'Sagar Navgire'

import random
import csv


# T - Number of Minority Class Samples
# N - RAte of Under-sampling

def underSample(T, N, majoritySamples, numberMajoritySamples):
    print ("In Under Sample")

    targetNumberMajoritySamples = int((T * 100) / N)
    print ("Number of target Majority Samples: " + str(targetNumberMajoritySamples))

    if (targetNumberMajoritySamples>=numberMajoritySamples):
        return majoritySamples

    copySamples = majoritySamples[:]

    while (numberMajoritySamples > targetNumberMajoritySamples):
        indexToDelete = random.randint(0, numberMajoritySamples - 1)
        #print indexToDelete
        copySamples.pop(indexToDelete)
        numberMajoritySamples -= 1

    print ("Number of Majority class: " + str(len(copySamples)))


    return copySamples
