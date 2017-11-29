__author__ = 'Sagar Navgire'

import csv
from smote import smote
from SmotePhoneme import smoteforPhoneme
from SmoteSatimage import smoteforSatimage
from SmoteMamography import smoteforMamography
from UnderSample import underSample
from NaiveBayes import naiveBayes
from DecisionTree import decisionTree

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial import ConvexHull
import numpy as np



if __name__ == '__main__':
    while (1):

        print ("Enter the dataset number to view results:")
        print ("1. PIMA - Indian diabetes dataset")
        print ("2. Phoneme dataset")
        print ("3. Satimage dataset")
        print ("4. Mammography dataset")
        print ("5. Exit.")
        choice = input("Enter you choice: ")

        if (int(choice) == 1):

            try:
                minorityCounter = 0
                majorityCounter = 0
                minoritySamples = []
                majoritySamples = []

                #Parsing input data
                with open('Data/PIMA/Input/diabetes.csv', 'rb') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        Sample = row[0].split(',')
                        if (Sample[8] == '1'):
                            minorityCounter += 1
                            minoritySamples.append(Sample[0:9])
                        elif (Sample[8] == '0'):
                            majorityCounter += 1
                            majoritySamples.append(Sample[0:9])

                print ("Number of Minority Samples:" + str(minorityCounter))
                print ("Number of Majority Samples:" + str(majorityCounter))

                csvfile.close()

                # Creating sythetic datasets - Under sampling and SMOTE
                underSamplingRates = [10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800]
                for u in underSamplingRates:
                    print "----------------Obtaining files for Under Sampling Rate = "+ str(u)
                    filenameSMOTE = "Data/PIMA/Output/diabetes_SMOTE_" + str(u) + ".csv"
                    filenameUnder = "Data/PIMA/Output/diabetes_Under_" + str(u) + ".csv"

                    fSmote = open(filenameSMOTE, "w+")
                    fUnder = open(filenameUnder, "w+")

                    for samples in minoritySamples:
                        fUnder.write((',').join(l for l in samples)+'\n')
                    fUnder.close()

                    smote(minorityCounter, 100, 5, minoritySamples, fSmote)
                    fSmote.close()

                    fSmote = open(filenameSMOTE, "a")
                    fUnder = open(filenameUnder, "a")


                    underSampledMajoritySamples = underSample(minorityCounter, u, majoritySamples, majorityCounter)
                    for samples in underSampledMajoritySamples:
                        fSmote.write((',').join(l for l in samples)+'\n')
                        fUnder.write((',').join(l for l in samples)+'\n')

                    fSmote.close()
                    fUnder.close()

                #Naive Bayes
                nbmean_fpr, nbmean_tpr = naiveBayes('Data/PIMA/Input/diabetes.csv', 8)
                nbmean_auc = auc(nbmean_fpr, nbmean_tpr, reorder=True) / 100
                plt.plot(nbmean_fpr, nbmean_tpr, color='b', label=r'Naive Bayes ROC (AUC = %0.2f)' % (nbmean_auc), lw=2, alpha=.8)

                # C4.5 Decision Tree - SMOTE
                files = ['Data/PIMA/Input/diabetes.csv','Data/PIMA/Output/diabetes_SMOTE_10.csv', 'Data/PIMA/Output/diabetes_SMOTE_15.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_20.csv','Data/PIMA/Output/diabetes_SMOTE_25.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_50.csv', 'Data/PIMA/Output/diabetes_SMOTE_75.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_100.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_125.csv', 'Data/PIMA/Output/diabetes_SMOTE_150.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_175.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_200.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_300.csv', 'Data/PIMA/Output/diabetes_SMOTE_400.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_500.csv',
                         'Data/PIMA/Output/diabetes_SMOTE_800.csv']

                smotemean_fpr, smotemean_tpr = decisionTree(files=files, numberOfFeatures=8)
                smotemean_auc = auc(smotemean_fpr, smotemean_tpr, reorder=True) / 100
                plt.plot(smotemean_fpr, smotemean_tpr, color='r',
                         label=r'C4.5 SMOTE + Under ROC (AUC = %0.2f)' % (smotemean_auc), lw=2, alpha=.8)
                # plt.plot([0, 100], [0, 100], linestyle='--', lw=2, color='r',
                #          label='Guessing', alpha=.8)

                # C4.5 Decision Tree - Under
                files = ['Data/PIMA/Input/diabetes.csv','Data/PIMA/Output/diabetes_Under_10.csv', 'Data/PIMA/Output/diabetes_Under_15.csv',
                         'Data/PIMA/Output/diabetes_Under_20.csv', 'Data/PIMA/Output/diabetes_Under_25.csv',
                         'Data/PIMA/Output/diabetes_Under_50.csv', 'Data/PIMA/Output/diabetes_Under_75.csv', 'Data/PIMA/Output/diabetes_Under_100.csv',
                         'Data/PIMA/Output/diabetes_Under_125.csv', 'Data/PIMA/Output/diabetes_Under_150.csv', 'Data/PIMA/Output/diabetes_Under_175.csv',
                         'Data/PIMA/Output/diabetes_Under_200.csv',
                         'Data/PIMA/Output/diabetes_Under_300.csv', 'Data/PIMA/Output/diabetes_Under_400.csv', 'Data/PIMA/Output/diabetes_Under_500.csv',
                         'Data/PIMA/Output/diabetes_Under_800.csv']

                undermean_fpr, undermean_tpr = decisionTree(files=files, numberOfFeatures=8)
                # print ("Under fprs: " + str(undermean_fpr))
                # print ("Under tprs: " + str(undermean_tpr))

                undermean_auc = auc(undermean_fpr, undermean_tpr, reorder=True) / 100
                plt.plot(undermean_fpr, undermean_tpr, color='g', label=r'C4.5 Under Sampling ROC (AUC = %0.2f)' % (undermean_auc), lw=2, alpha=.8)



                # Convex Hull
                fprs = undermean_fpr.tolist() + smotemean_fpr.tolist() + nbmean_fpr
                tprs = undermean_tpr.tolist() + smotemean_tpr.tolist() + nbmean_tpr

                points = np.column_stack((fprs, tprs))
                hull = ConvexHull(points)
                # plt.plot(points[:, 0], points[:, 1], 'o')
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k--')

                # xy = np.hstack(fprs[:, np.newaxis], tprs[:, np.newaxis])
                # hull = ConvexHull(xy)
                # plt.scatter(fprs, tprs)
                # plt.plot(xy[hull.vertices], xy[hull.vertices])

                plt.xlim([0, 105])
                plt.ylim([50, 105])
                plt.xlabel('% False Positive')
                plt.ylabel('% True Positive')
                plt.title('PIMA Receiver operating characteristics')
                plt.legend(loc="best")
                plt.savefig('Graphics/PIMA_ROC.png')
                plt.show()

            except Exception as error:
                print (error)


        elif (int(choice) == 2):
            print ("For Phoneme")

            phonemeMinorityCounter = 0
            phonemeMajorityCounter = 0
            phonemeMinoritySamples = []
            phonemeMajoritySamples = []

            try:
                #Parse and generate datasets for Phoneme
                for i in open("Data/Phoneme/Input/phoneme.dat").readlines():
                    Sample = i.strip().split()
                    if (Sample[5] == '1'):
                        phonemeMinorityCounter += 1
                        phonemeMinoritySamples.append(Sample[0:6])
                    elif (Sample[5] == '0'):
                        phonemeMajorityCounter += 1
                        phonemeMajoritySamples.append(Sample[0:6])

                print ("Number of phoneme Miniority Samples:" + str(phonemeMinorityCounter))
                print ("Number of phoneme Majority Samples:" + str(phonemeMajorityCounter))

                f = open("Data/Phoneme/Input/phoneme.csv", "w+")
                phonemeAllSamples = phonemeMajoritySamples + phonemeMinoritySamples
                for samples in phonemeAllSamples:
                    f.write((',').join(l for l in samples) + '\n')
                f.close()

                # Creating sythetic datasets - Under sampling and SMOTE
                # # underSamplingRates = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800]
                underSamplingRates = [10, 25, 50, 100, 200, 500, 800, 1000, 1500]
                for u in underSamplingRates:
                    print "----------------Obtaining files for Under Sampling Rate = " + str(u)
                    filenameSMOTE = "Data/Phoneme/Output/phoneme_SMOTE_" + str(u) + ".csv"
                    filenameUnder = "Data/Phoneme/Output/phoneme_Under_" + str(u) + ".csv"

                    fSmote = open(filenameSMOTE, "w+")
                    fUnder = open(filenameUnder, "w+")

                    for samples in phonemeMinoritySamples:
                        fUnder.write((',').join(l for l in samples) + '\n')
                    fUnder.close()

                    smoteforPhoneme(phonemeMinorityCounter, 200, 5, phonemeMinoritySamples, fSmote)
                    fSmote.close()

                    fSmote = open(filenameSMOTE, "a")
                    fUnder = open(filenameUnder, "a")

                    underSampledMajoritySamples = underSample(phonemeMinorityCounter, u, phonemeMajoritySamples, phonemeMajorityCounter)
                    for samples in underSampledMajoritySamples:
                        fSmote.write((',').join(l for l in samples) + '\n')
                        fUnder.write((',').join(l for l in samples) + '\n')

                    fSmote.close()
                    fUnder.close()

                # Naive Bayes
                nbmean_fpr, nbmean_tpr = naiveBayes('Data/Phoneme/Input/phoneme.csv', 5)
                nbmean_auc = auc(nbmean_fpr, nbmean_tpr, reorder=True) / 100
                plt.plot(nbmean_fpr, nbmean_tpr, color='b', label=r'Naive Bayes ROC (AUC = %0.2f)' % (nbmean_auc),
                             lw=2, alpha=.8)

                # C4.5 Decision Tree - SMOTE
                files = ['Data/Phoneme/Input/phoneme.csv', 'Data/Phoneme/Output/phoneme_SMOTE_10.csv',
                         # 'Data/Phoneme/Output/phoneme_SMOTE_15.csv',
                          'Data/Phoneme/Output/phoneme_SMOTE_25.csv','Data/Phoneme/Output/phoneme_SMOTE_50.csv',
                         # 'Data/Phoneme/Output/phoneme_SMOTE_75.csv',
                         'Data/Phoneme/Output/phoneme_SMOTE_100.csv',
                         # 'Data/Phoneme/Output/phoneme_SMOTE_125.csv', 'Data/Phoneme/Output/phoneme_SMOTE_150.csv',
                         # 'Data/Phoneme/Output/phoneme_SMOTE_175.csv',
                         'Data/Phoneme/Output/phoneme_SMOTE_200.csv',
                         # 'Data/Phoneme/Output/phoneme_SMOTE_300.csv', 'Data/Phoneme/Output/phoneme_SMOTE_400.csv',
                         'Data/Phoneme/Output/phoneme_SMOTE_500.csv',
                         'Data/Phoneme/Output/phoneme_SMOTE_800.csv', 'Data/Phoneme/Output/phoneme_SMOTE_1000.csv',
                         'Data/Phoneme/Output/phoneme_SMOTE_1500.csv']

                smotemean_fpr, smotemean_tpr = decisionTree(files=files, numberOfFeatures=5)
                smotemean_auc = auc(smotemean_fpr, smotemean_tpr, reorder=True) / 100
                plt.plot(smotemean_fpr, smotemean_tpr, color='g',
                         label=r'C4.5 SMOTE + Under ROC (AUC = %0.2f)' % (smotemean_auc), lw=2, alpha=.8)

                # C4.5 Decision Tree - Under
                files = ['Data/Phoneme/Input/phoneme.csv','Data/Phoneme/Output/phoneme_Under_10.csv', #'Data/Phoneme/Output/phoneme_Under_15.csv',
                         'Data/Phoneme/Output/phoneme_Under_25.csv',
                         'Data/Phoneme/Output/phoneme_Under_50.csv',
                         # 'Data/Phoneme/Output/phoneme_Under_75.csv',
                         'Data/Phoneme/Output/phoneme_Under_100.csv',
                         # 'Data/Phoneme/Output/phoneme_Under_125.csv', 'Data/Phoneme/Output/phoneme_Under_150.csv',
                         # 'Data/Phoneme/Output/phoneme_Under_175.csv',
                         'Data/Phoneme/Output/phoneme_Under_200.csv',
                         # 'Data/Phoneme/Output/phoneme_Under_300.csv', 'Data/Phoneme/Output/phoneme_Under_400.csv',
                         'Data/Phoneme/Output/phoneme_Under_500.csv',
                         'Data/Phoneme/Output/phoneme_Under_800.csv', 'Data/Phoneme/Output/phoneme_Under_1000.csv',
                         'Data/Phoneme/Output/phoneme_Under_1500.csv']

                undermean_fpr, undermean_tpr = decisionTree(files=files, numberOfFeatures=5)
                # print ("Under fprs: " + str(undermean_fpr))
                # print ("Under tprs: " + str(undermean_tpr))

                undermean_auc = auc(undermean_fpr, undermean_tpr, reorder=True) / 100
                plt.plot(undermean_fpr, undermean_tpr, color='r',
                         label=r'C4.5 Under Sampling ROC (AUC = %0.2f)' % (undermean_auc), lw=2, alpha=.8)



                # Convex Hull
                fprs = undermean_fpr.tolist() + smotemean_fpr.tolist() + nbmean_fpr
                tprs = undermean_tpr.tolist() + smotemean_tpr.tolist() + nbmean_tpr

                points = np.column_stack((fprs, tprs))
                hull = ConvexHull(points)
                # plt.plot(points[:, 0], points[:, 1], 'o')
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k--')

                plt.xlim([0, 105])
                plt.ylim([50, 105])
                plt.xlabel('% False Positive')
                plt.ylabel('% True Positive')
                plt.title('Phoneme Receiver operating characteristics')
                plt.legend(loc="best")
                plt.savefig('Graphics/Phoneme_ROC.png')
                plt.show()



            except Exception as error:
                print (error)



        elif (int(choice) == 3):
            print ("For Satimage")

            satimageMinorityCounter = 0
            satimageMajorityCounter = 0
            satimageMinoritySamples = []
            satimageMajoritySamples = []

            try:
                #Parse and generate datasets for Satimage
                for i in open("Data/Satimage/Input/satimage.dat").readlines():
                    Sample = i.strip().split()
                    # Label '4' is minority (+ve) class
                    if (Sample[36] == '4'):
                        satimageMinorityCounter += 1
                        Sample[36] = '1'
                        satimageMinoritySamples.append(Sample[0:37])
                    else:
                        satimageMajorityCounter += 1
                        Sample[36] = '0'
                        satimageMajoritySamples.append(Sample[0:37])

                print ("Number of satimage Miniority Samples:" + str(satimageMinorityCounter))
                print ("Number of satimage Majority Samples:" + str(satimageMajorityCounter))

                f = open("Data/Satimage/Input/satimage.csv", "w+")
                satimageAllSamples = satimageMajoritySamples + satimageMinoritySamples
                for samples in satimageAllSamples:
                    f.write((',').join(l for l in samples) + '\n')
                f.close()

                # Creating sythetic datasets - Under sampling and SMOTE
                underSamplingRates = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000]
                # underSamplingRates = [10, 100, 500, 800, 1000, 1500, 2000]
                for u in underSamplingRates:
                    print "----------------Obtaining files for Under Sampling Rate = " + str(u)
                    filenameSMOTE = "Data/Satimage/Output/satimage_SMOTE_" + str(u) + ".csv"
                    filenameUnder = "Data/Satimage/Output/satimage_Under_" + str(u) + ".csv"

                    fSmote = open(filenameSMOTE, "w+")
                    fUnder = open(filenameUnder, "w+")

                    for samples in satimageMinoritySamples:
                        fUnder.write((',').join(l for l in samples) + '\n')
                    fUnder.close()

                    smoteforSatimage(satimageMinorityCounter, 200, 5, satimageMinoritySamples, fSmote)
                    fSmote.close()

                    fSmote = open(filenameSMOTE, "a")
                    fUnder = open(filenameUnder, "a")

                    underSampledMajoritySamples = underSample(satimageMinorityCounter, u, satimageMajoritySamples, satimageMajorityCounter)
                    for samples in underSampledMajoritySamples:
                        fSmote.write((',').join(l for l in samples) + '\n')
                        fUnder.write((',').join(l for l in samples) + '\n')

                    fSmote.close()
                    fUnder.close()

                # Naive Bayes
                nbmean_fpr, nbmean_tpr = naiveBayes('Data/Satimage/Input/satimage.csv', 36)
                nbmean_auc = auc(nbmean_fpr, nbmean_tpr, reorder=True) / 100
                plt.plot(nbmean_fpr, nbmean_tpr, color='b', label=r'Naive Bayes ROC (AUC = %0.2f)' % (nbmean_auc),
                         lw=2, alpha=.8)

                # C4.5 Decision Tree - SMOTE
                files = ['Data/Satimage/Input/satimage.csv','Data/Satimage/Output/satimage_SMOTE_10.csv', 'Data/Satimage/Output/satimage_SMOTE_15.csv',
                         'Data/Satimage/Output/satimage_SMOTE_25.csv',
                         'Data/Satimage/Output/satimage_SMOTE_50.csv', 'Data/Satimage/Output/satimage_SMOTE_75.csv',
                         'Data/Satimage/Output/satimage_SMOTE_100.csv',
                         'Data/Satimage/Output/satimage_SMOTE_125.csv', 'Data/Satimage/Output/satimage_SMOTE_150.csv',
                         'Data/Satimage/Output/satimage_SMOTE_175.csv',
                         'Data/Satimage/Output/satimage_SMOTE_200.csv',
                         'Data/Satimage/Output/satimage_SMOTE_300.csv', 'Data/Satimage/Output/satimage_SMOTE_400.csv',
                         'Data/Satimage/Output/satimage_SMOTE_500.csv',
                         'Data/Satimage/Output/satimage_SMOTE_800.csv', 'Data/Satimage/Output/satimage_SMOTE_1000.csv',
                         'Data/Satimage/Output/satimage_SMOTE_1500.csv', 'Data/Satimage/Output/satimage_SMOTE_2000.csv']

                smotemean_fpr, smotemean_tpr = decisionTree(files=files, numberOfFeatures=36)
                smotemean_auc = auc(smotemean_fpr, smotemean_tpr, reorder=True) / 100
                plt.plot(smotemean_fpr, smotemean_tpr, color='g',
                         label=r'C4.5 SMOTE + Under ROC (AUC = %0.2f)' % (smotemean_auc), lw=2, alpha=.8)

                # C4.5 Decision Tree - Under
                files = ['Data/Satimage/Input/satimage.csv','Data/Satimage/Output/satimage_Under_10.csv', 'Data/Satimage/Output/satimage_Under_15.csv',
                         'Data/Satimage/Output/satimage_Under_25.csv',
                         'Data/Satimage/Output/satimage_Under_50.csv', 'Data/Satimage/Output/satimage_Under_75.csv',
                         'Data/Satimage/Output/satimage_Under_100.csv',
                         'Data/Satimage/Output/satimage_Under_125.csv', 'Data/Satimage/Output/satimage_Under_150.csv',
                         'Data/Satimage/Output/satimage_Under_175.csv',
                         'Data/Satimage/Output/satimage_Under_200.csv',
                         'Data/Satimage/Output/satimage_Under_300.csv', 'Data/Satimage/Output/satimage_Under_400.csv',
                         'Data/Satimage/Output/satimage_Under_500.csv',
                         'Data/Satimage/Output/satimage_Under_800.csv','Data/Satimage/Output/satimage_Under_1000.csv',
                         'Data/Satimage/Output/satimage_Under_1500.csv',
                         'Data/Satimage/Output/satimage_Under_2000.csv']

                undermean_fpr, undermean_tpr = decisionTree(files=files, numberOfFeatures=36)
                # print ("Under fprs: " + str(undermean_fpr))
                # print ("Under tprs: " + str(undermean_tpr))

                undermean_auc = auc(undermean_fpr, undermean_tpr, reorder=True) / 100
                plt.plot(undermean_fpr, undermean_tpr, color='r',
                         label=r'C4.5 Under Sampling ROC (AUC = %0.2f)' % (undermean_auc), lw=2, alpha=.8)


                # Convex Hull
                fprs = undermean_fpr.tolist() + smotemean_fpr.tolist() + nbmean_fpr
                tprs = undermean_tpr.tolist() + smotemean_tpr.tolist() + nbmean_tpr

                points = np.column_stack((fprs, tprs))
                hull = ConvexHull(points)
                # plt.plot(points[:, 0], points[:, 1], 'o')
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k--')

                plt.xlim([0, 105])
                plt.ylim([50, 105])
                plt.xlabel('% False Positive')
                plt.ylabel('% True Positive')
                plt.title('Satimage Receiver operating characteristics')
                plt.legend(loc="best")
                plt.savefig('Graphics/Satimage_ROC.png')
                plt.show()


            except Exception as error:
                print (error)

        elif (int(choice) == 4):
            mamMinorityCounter = 0
            mamMajorityCounter = 0
            mamMinoritySamples = []
            mamMajoritySamples = []

            try:
                # Parsing input data
                with open('Data/Mamography/Input/mammography.csv', 'rb') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        Sample = row[0].split(',')
                        if (Sample[6] == '1.0'):
                            mamMinorityCounter += 1
                            mamMinoritySamples.append(Sample[0:7])
                        elif (Sample[6] == '0.0'):
                            mamMajorityCounter += 1
                            mamMajoritySamples.append(Sample[0:7])

                print ("Number of Minority Samples:" + str(mamMinorityCounter))
                print ("Number of Majority Samples:" + str(mamMajorityCounter))

                # Creating sythetic datasets - Under sampling and SMOTE
                underSamplingRates = [10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
                # underSamplingRates = [10, 50, 100, 500, 800, 1000]
                for u in underSamplingRates:
                    print "----------------Obtaining files for Under Sampling Rate = " + str(u)
                    filenameSMOTE = "Data/Mamography/Output/mammography_SMOTE_" + str(u) + ".csv"
                    filenameUnder = "Data/Mamography/Output/mammography_Under_" + str(u) + ".csv"

                    fSmote = open(filenameSMOTE, "w+")
                    fUnder = open(filenameUnder, "w+")

                    for samples in mamMinoritySamples:
                        fUnder.write((',').join(l for l in samples) + '\n')
                    fUnder.close()

                    smoteforMamography(mamMinorityCounter, 400, 5, mamMinoritySamples, fSmote)
                    fSmote.close()

                    fSmote = open(filenameSMOTE, "a")
                    fUnder = open(filenameUnder, "a")

                    underSampledMajoritySamples = underSample(mamMinorityCounter, u, mamMajoritySamples, mamMajorityCounter)
                    for samples in underSampledMajoritySamples:
                        fSmote.write((',').join(l for l in samples) + '\n')
                        fUnder.write((',').join(l for l in samples) + '\n')

                    fSmote.close()
                    fUnder.close()

                # Naive Bayes
                nbmean_fpr, nbmean_tpr = naiveBayes('Data/Mamography/Input/mammography.csv', 6)
                nbmean_auc = auc(nbmean_fpr, nbmean_tpr, reorder=True) / 100
                plt.plot(nbmean_fpr, nbmean_tpr, color='b', label=r'Naive Bayes ROC (AUC = %0.2f)' % (nbmean_auc), lw=2,
                         alpha=.8)

                # C4.5 Decision Tree - SMOTE
                files = ['Data/Mamography/Input/mammography.csv', 'Data/Mamography/Output/mammography_SMOTE_10.csv',
                         'Data/Mamography/Output/mammography_SMOTE_15.csv',
                         'Data/Mamography/Output/mammography_SMOTE_20.csv', 'Data/Mamography/Output/mammography_SMOTE_25.csv',
                         'Data/Mamography/Output/mammography_SMOTE_50.csv', 'Data/Mamography/Output/mammography_SMOTE_75.csv',
                         'Data/Mamography/Output/mammography_SMOTE_100.csv',
                         'Data/Mamography/Output/mammography_SMOTE_125.csv', 'Data/Mamography/Output/mammography_SMOTE_150.csv',
                         'Data/Mamography/Output/mammography_SMOTE_175.csv',
                         'Data/Mamography/Output/mammography_SMOTE_200.csv',
                         'Data/Mamography/Output/mammography_SMOTE_300.csv', 'Data/Mamography/Output/mammography_SMOTE_400.csv',
                         'Data/Mamography/Output/mammography_SMOTE_500.csv']
                         # 'Data/Mamography/Output/mammography_SMOTE_800.csv',] #'Data/Mamography/Output/mammography_SMOTE_1000.csv']

                smotemean_fpr, smotemean_tpr = decisionTree(files=files, numberOfFeatures=6)
                smotemean_auc = auc(smotemean_fpr, smotemean_tpr, reorder=True) / 100
                plt.plot(smotemean_fpr, smotemean_tpr, color='r',
                         label=r'C4.5 SMOTE + Under ROC (AUC = %0.2f)' % (smotemean_auc), lw=2, alpha=.8)
                # plt.plot([0, 100], [0, 100], linestyle='--', lw=2, color='r',
                #          label='Guessing', alpha=.8)

                # C4.5 Decision Tree - Under
                files = ['Data/Mamography/Input/mammography.csv', 'Data/Mamography/Output/mammography_Under_10.csv',
                         'Data/Mamography/Output/mammography_Under_15.csv',
                         'Data/Mamography/Output/mammography_Under_20.csv', 'Data/Mamography/Output/mammography_Under_25.csv',
                         'Data/Mamography/Output/mammography_Under_50.csv', 'Data/Mamography/Output/mammography_Under_75.csv',
                         'Data/Mamography/Output/mammography_Under_100.csv',
                         'Data/Mamography/Output/mammography_Under_125.csv', 'Data/Mamography/Output/mammography_Under_150.csv',
                         'Data/Mamography/Output/mammography_Under_175.csv',
                         'Data/Mamography/Output/mammography_Under_200.csv',
                         'Data/Mamography/Output/mammography_Under_300.csv', 'Data/Mamography/Output/mammography_Under_400.csv',
                         'Data/Mamography/Output/mammography_Under_500.csv']
                         # 'Data/Mamography/Output/mammography_Under_800.csv',]# 'Data/Mamography/Output/mammography_Under_1000.csv']

                undermean_fpr, undermean_tpr = decisionTree(files=files, numberOfFeatures=6)
                # print ("Under fprs: " + str(undermean_fpr))
                # print ("Under tprs: " + str(undermean_tpr))

                undermean_auc = auc(undermean_fpr, undermean_tpr, reorder=True) / 100
                plt.plot(undermean_fpr, undermean_tpr, color='g',
                         label=r'C4.5 Under Sampling ROC (AUC = %0.2f)' % (undermean_auc), lw=2, alpha=.8)

                # Convex Hull
                fprs = undermean_fpr.tolist() + smotemean_fpr.tolist() + nbmean_fpr
                tprs = undermean_tpr.tolist() + smotemean_tpr.tolist() + nbmean_tpr

                points = np.column_stack((fprs, tprs))
                hull = ConvexHull(points)
                # plt.plot(points[:, 0], points[:, 1], 'o')
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k--')

                plt.xlim([0, 105])
                plt.ylim([50, 105])
                plt.xlabel('% False Positive')
                plt.ylabel('% True Positive')
                plt.title('Mamography Receiver operating characteristics')
                plt.legend(loc="best")
                plt.savefig('Graphics/Mammography_ROC.png')
                plt.show()

                csvfile.close()

            except Exception as error:
                print (error)


        else:
            exit(1)
