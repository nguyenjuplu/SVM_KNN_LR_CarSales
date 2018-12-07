import sys

def printResult(predictingSample, predictingTarget, actualTarget):
    """
    print out the sale prediction and its corresponding month and year,
    and the accurracy comparing to the real values.
    """
    total = 0.0
    size = len(predictingSample)
    print("Sale prediction in {} months: ".format(size))
    print("{}\t{}\t\t{}\t\t{}".format("Mth,Yr", "Sale prediction", "Actual Value", "%"))
    for i in range(size):
        percent = 0.0
        if(predictingTarget[i] < actualTarget[i]):
            percent = predictingTarget[i] / actualTarget[i]
        else:    
            percent = actualTarget[i] / predictingTarget[i] 
        print("{}  {}\t\t\t{}\t\t{}".format(predictingSample[i], predictingTarget[i], actualTarget[i], percent))
        total += percent
    accuracy = total/size

    print("Accuracy between the prediction and the actual sales is: ", accuracy)
    return accuracy

def svrPrediction(sampleList, targetList, predictingSampleList):
    """
        The svr_rbf prediction function that takes in 2D sample list and 1D target list, and return the 1D predicting list.
    """
    from sklearn import svm
    X = sampleList
    y = targetList
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma = 'scale')
    y_rbf = svr_rbf.fit(X, y).predict(predictingSampleList)
    y_rbf = [int(round(x)) for x in y_rbf]
    return y_rbf

def extractData(sampleFile):
    """
    Given input file, return a 2D list of the sample represeting the month and year and a 1D of the target represeting the sale. 
    """
    file = open(sampleFile, 'r')
    #skipping heading
    file.readline()
    sampleList = []
    targetList = []
    month = 1
    yr = 7
    for line in file:
        line = line.split()
        targetList.append( int(line[0]) )
        if month == 13:
            month = 1   
            yr +=1
        sampleList.append([month, yr])
        month +=1
    file.close()    
    return sampleList, targetList

def main():
    dataSet, target = extractData(sys.argv[1])
    k=12
    total = 0.0
    rank = []
    for i in range(0,12): 
        start = i*12
        end = start + 12
        trainingSample = dataSet[:start]
        trainingSample += dataSet[end:]
        trainingTarget= target[:start]
        trainingTarget += target[end:]
        predictingSample = dataSet[start:end]
        testingActualTarget = target[start:end]
        rbf =  svrPrediction(trainingSample, trainingTarget, predictingSample)
        result = printResult(predictingSample, rbf, testingActualTarget)
        total += result  

    print("****Average of K=12 fold cross validation= ", total/k)


if __name__ == "__main__":
    main()