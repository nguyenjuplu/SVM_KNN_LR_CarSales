import sys
from operator import itemgetter

#Prints the results of each months predictions, and gets the average for the year, which is the fold
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

#Trains the neural network with the given training dataset and the actual values specified by each fold
def nnPrediction(sampleList, targetList, predictingSampleList):
    from sklearn.neural_network import MLPRegressor
    X = sampleList
    y = targetList

    ####
    nn = MLPRegressor(hidden_layer_sizes=(200, ), activation='logistic', solver='lbfgs', max_iter=400, alpha=5)
    y_nn = nn.fit(X, y).predict(predictingSampleList)
    y_nn = [int(round(x)) for x in y_nn]
    ###

    return y_nn

#Reads the input demand dataset and converts it into a 2d array and the y values for demand unit sales
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

#Trains the neural network for each fold, which is 1 year, and then prints the average of all 12 fold results
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
        model =  nnPrediction(trainingSample, trainingTarget, predictingSample)
        result = printResult(predictingSample, model, testingActualTarget)
        total += result
        rank.append([i, result])


    print("****Average of K=12 fold cross validation= ", total/k)


if __name__ == "__main__":
    main()