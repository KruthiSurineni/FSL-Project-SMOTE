
# Naive Bayes Classifier

import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import KFold
from decimal import getcontext, Decimal


def naiveBayes(filename, numberOfFeatures):
    features = []
    getcontext().prec = 3
    colors = {1: 'b', 2: 'g', 5: 'r', 10: 'c', 25: 'm', 50: 'y'}
    # with open('Input/diabetes.csv', 'rb') as csvDataFile:
    with open(filename, 'rb') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            features.append(map(float, row))
        data = np.array(features)[:, :numberOfFeatures]
        n_samples, n_features = data.shape
        target = np.array(map(int, np.array(features)[:, numberOfFeatures]))

        mean_fpr = []
        mean_tpr = []
        priorCombinations = []
        #  [ 0.65104167, 0.34895833], [0.5,0.5], [0.4,0.6],[0.25,0.75],[0.10,0.90],[0.05,0.95]]


        majorityDist = float(500)
        multiplier = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for m in multiplier:
            minorityDist = float(m) * majorityDist
            totalDist = minorityDist + majorityDist
            minority = float(minorityDist / totalDist)
            priorCombinations.append([1 - minority, minority])
        # print priorCombinations

        for prior in priorCombinations:
            gnb = GaussianNB(priors=prior)
            cv = KFold(n_splits=10, shuffle=True)
            tprs = []
            aucs = []
            fprs = []

            i = 0
            for train, test in cv.split(data):
                model = gnb.fit(data[train], target[train])
                y_pred = model.predict(data)
                fpr, tpr, threshold = metrics.roc_curve(target, y_pred)
                tprs.append(tpr[1] * 100)
                fprs.append(fpr[1] * 100)

                i += 1

            mean_tpr.append(float(sum(tprs)) / len(tprs))
            mean_fpr.append(float(sum(fprs)) / len(fprs))

        mean_fpr.append(100)
        mean_tpr.append(100)
        print mean_tpr
        print mean_fpr

    csvDataFile.close()
    return mean_fpr, mean_tpr
