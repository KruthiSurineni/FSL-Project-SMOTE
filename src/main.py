__author__ = 'Sagar Navgire'

import csv
from smote import smote
from UnderSample import underSample



if __name__ == '__main__':
    try:
        minorityCounter = 0
        majorityCounter = 0
        minoritySamples = []
        majoritySamples = []


        with open('Input/diabetes.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                Sample = row[0].split(',')
                if (Sample[8] == '1'):
                     minorityCounter += 1
                     minoritySamples.append(Sample[0:9])
            # print (Sample[0:8])

            # print(row[0][2])
            # print (', '.join(row))
                elif (Sample[8] == '0'):
                    majorityCounter += 1
                    majoritySamples.append(Sample[0:9])

        print ("Number of Minority Samples:" + str(minorityCounter))
        print ("Number of Majority Samples:" + str(majorityCounter))

        csvfile.close()

        underSamplingRates = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800]
        for u in underSamplingRates:
            print "----------------Obtaining files for Under Sampling Rate = "+ str(u)
            filenameSMOTE = "Output/diabetes_SMOTE_" + str(u) + ".csv"
            filenameUnder = "Output/diabetes_Under_" + str(u) + ".csv"

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







    except Exception as error:
        print (error)
