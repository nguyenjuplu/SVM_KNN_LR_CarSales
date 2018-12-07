import sys

def printResult(predictingSample, predictingTarget, actualTarget):
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

def knnPrediction(sampleList, targetList, predictingSampleList):
    from sklearn.neighbors import KNeighborsRegressor
    x = sampleList
    y = targetList
    neigh = KNeighborsRegressor(n_neighbors = 3)
    y_neigh = neigh.fit(x, y).predict(predictingSampleList)
    y_neigh = [int(round(x)) for x in y_neigh]
    return y_neigh

def extractData(sampleFile):
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
        knn =  knnPrediction(trainingSample, trainingTarget, predictingSample)
        result = printResult(predictingSample, knn, testingActualTarget)
        total += result


    print("****Average of K=12 fold cross validation = ", total/k)


if __name__ == "__main__":
    main()