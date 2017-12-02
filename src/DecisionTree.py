
# Decision Tree Classifier

import numpy as np
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier


def decisionTree(files, numberOfFeatures):
    features = []
    mean_tpr = []
    mean_fpr = []

    for file in files:
        print ("C4.5 in file: " + str(file))
        with open(file, 'r') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                features.append(map(float, row))
                data = np.array(features)[:, :numberOfFeatures]
            n_samples, n_features = data.shape
            target = np.array(map(int, np.array(features)[:, numberOfFeatures]))

            clf = DecisionTreeClassifier(random_state=0)
            cv = KFold(n_splits=10, shuffle=True)
            tprs = []
            fprs = []

            i = 0
            for train, test in cv.split(data):
                model = clf.fit(data[train], target[train])
                probas = model.predict(data[test])
                fpr, tpr, thresholds = roc_curve(target[test], probas)
                tprs.append(tpr[1] * 100)
                fprs.append(fpr[1] * 100)
                i += 1

            mean_tpr.append(float(sum(tprs)) / len(tprs))
            mean_fpr.append(float(sum(fprs)) / len(fprs))

    mean_tpr.append(100)
    mean_fpr.append(100)

    mean_tpr = np.sort(mean_tpr)
    mean_fpr = np.sort(mean_fpr)

    print mean_tpr
    print mean_fpr

    csvDataFile.close()
    return mean_fpr, mean_tpr
