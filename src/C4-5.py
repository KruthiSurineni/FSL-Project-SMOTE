# Compute ROC curve and area the curve
print(__doc__)
import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

features = []
mean_tpr=[]
mean_fpr=[]

files = ['Output/diabetes_SMOTE_10.csv','Output/diabetes_SMOTE_15.csv','Output/diabetes_SMOTE_25.csv','Output/diabetes_SMOTE_50.csv','Output/diabetes_SMOTE_75.csv','Output/diabetes_SMOTE_100.csv','Output/diabetes_SMOTE_125.csv','Output/diabetes_SMOTE_150.csv','Output/diabetes_SMOTE_175.csv','Output/diabetes_SMOTE_200.csv',
         'Output/diabetes_SMOTE_300.csv','Output/diabetes_SMOTE_400.csv','Output/diabetes_SMOTE_500.csv','Output/diabetes_SMOTE_800.csv']
for file in files:
    with open(file, 'r') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            features.append(map(float, row))
            data = np.array(features)[:, :8]
        n_samples, n_features = data.shape
        target = np.array(map(int, np.array(features)[:, 8]))

        clf = DecisionTreeClassifier(random_state=0)
        cv = KFold(n_splits=10, shuffle=True)
        tprs = []
        fprs = []

        #mean_fpr = np.linspace(0, 100, 10)

        i = 0
        for train, test in cv.split(data):
            model = clf.fit(data[train], target[train])
            #print data[test]
            probas = model.predict(data[test])
            fpr, tpr, thresholds = roc_curve(target[test], probas)
            tprs.append(tpr[1] * 100)
            fprs.append(fpr[1] * 100)
            i += 1




        mean_tpr.append(float(sum(tprs)) / len(tprs))
        mean_fpr.append(float(sum(fprs)) / len(fprs))

mean_tpr = np.sort(mean_tpr)
mean_fpr = np.sort(mean_fpr)
np.append(mean_tpr,100)
np.append(mean_fpr,100)

print mean_tpr
print mean_fpr
mean_auc = auc(mean_fpr, mean_tpr, reorder=True) / 100


plt.plot(mean_fpr, mean_tpr, color='b', label=r'C4.5 ROC (AUC = %0.2f)' % (mean_auc), lw=2, alpha=.8)
plt.plot([0, 100], [0, 100], linestyle='--', lw=2, color='r',
            label='Guessing', alpha=.8)
csvDataFile.close()
plt.xlim([15, 105])
plt.ylim([50, 105])
plt.xlabel('% False Positive')
plt.ylabel('% True Positive')
plt.title('PIMA Receiver operating characteristics')
plt.legend(loc="best")
plt.show()
