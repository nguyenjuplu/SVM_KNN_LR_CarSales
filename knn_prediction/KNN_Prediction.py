import sys

def main():

    #################################################
    #1) Check Input File                            #
    #################################################
    if(not CheckInput(sys.argv)):
        sys.exit(0)
    inputFile = sys.argv[1]

    #################################################
    #2) Get dataSet into month and year             #
    #   Target is corresponding sale                #
    #################################################
    dataSet, target = extractData(inputFile)

    #########################################################
    #3) Get user input for model training and prediction    #
    #   Variables:                                          #
    #       frmMth, frmYr                                   #
    #       toMth, toYr                                     #
    #########################################################
    print("Dataset: {01-2007 to 10-2018}")
    print("Enter desired length for model training (Example: 01 2007 01 2008)")
    userInput = input("From month_year to month_year:")
    userInput = userInput.split()
    if len(userInput) != 4:
        print("Incorrect input, try again")
        userInput = input("From month_year to month_year:")
        userInput = userInput.split()

    frmMth, frmYr, toMth, toYr = [int(x) for x in userInput]

    nextMths = input("How many months ahead do you want to predict? Standard is 12")
    nextMths = int(nextMths)

    #####################################################################
    #4) Split data to training sample, training target,                 #
    #                prediction sample, and optional testing target     #
    #####################################################################
    trainingSample, trainingTarget, predictingSample, testingTarget = \
        splitDataSet(dataSet, target, frmMth, frmYr, toMth, toYr, nextMths, "test-yes")

    #trainingModelTarget is for graph to show the trace of the training model before prediction
    predictingTarget, trainingModelTarget = knnPrediction(trainingSample, trainingTarget, predictingSample)


    #####################################################
    #5) Compare Prediction sale and Actual Sale         #
    #       Print the predicting result and accuracy    #
    #####################################################
    printResult(predictingSample, predictingTarget, testingTarget)

    #6) Plot the training graph, prediction graph, and the trace of training model
    plotGraph(trainingSample, trainingTarget, predictingSample, predictingTarget, trainingModelTarget)



def CheckInput(argvs):
    if len(argvs) != 2:
        print("Input dataset textfile is not found.")
        return False
    return True


def extractData(inputFile):
    sampleList = []
    targetList = []
    month = 1
    year = 7
    file = open(inputFile, 'r')
    file.readline()  # Skip Header

    #Read File and translate data
    for line in file:
        line = line.split()
        targetList.append(int(line[0]))
        if month == 13:
            month = 1
            year += 1
        sampleList.append([month, year])
        month += 1

    file.close()

    return sampleList, targetList

def splitDataSet(sampleList, targetList, fromMonth, fromYear, toMonth, toYear, nextMonth, testingFlag):
    numSample = (toYear - fromYear + 1) * 12 - (fromMonth - 1) - (12 - toMonth)
    fromIndex = (fromYear - 7) * 12 + (fromMonth - 1)
    toIndex = fromIndex + numSample

    trainingSample = sampleList[fromIndex:toIndex]
    trainingTarget = targetList[fromIndex:toIndex]

    month = toMonth
    year = toYear
    predictSample = []

    for i in range(nextMonth):
        month += 1
        if (month == 13):
            month = 1
            year += 1
        predictSample.append([month, year])

    if testingFlag == "test-yes":
        testTarget = targetList[toIndex:toIndex + nextMonth]
        return trainingSample, trainingTarget, predictSample, testTarget
    else:
        return trainingSample, trainingTarget, predictSample


def knnPrediction(sampleList, targetList, predictingSampleList):
    from sklearn.neighbors import KNeighborsRegressor
    x = sampleList
    y = targetList
    neigh = KNeighborsRegressor(n_neighbors = 3)
    print(sampleList)
    print(targetList)
    print(predictingSampleList)
    y_neigh = neigh.fit(x, y).predict(predictingSampleList)
    y_neigh = [int(round(x)) for x in y_neigh]

    #Use for graping to show trade of training model
    y_modelTraining = neigh.fit(x, y).predict(sampleList)

    return y_neigh, y_modelTraining

def plotGraph(trainingSample, trainingTarget, predictingSample, predictingTarget, trainingModelTarget):
    import matplotlib.pyplot as plt
    lw = 2
    startMth = trainingSample[0][0]
    xSample = [i+startMth for i in range(len(trainingSample))]
    xPredict = [i+1 for i in range(xSample[-1], xSample[-1] + len(predictingSample))]
    plt.scatter(xSample, trainingTarget, color='darkorange', label='Data Sample')
    plt.plot(xSample, trainingModelTarget, color='c', lw=lw, label='KNN model')
    plt.scatter(xPredict, predictingTarget, color='navy', lw=lw, label='KNN prediction')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


def printResult(predictingSample, predictingTarget, testingTarget):
    total = 0.0
    size = len(predictingSample)
    print("Sale Prediction in {} months: ".format(size))
    print("{}\t\t{}\t{}\t{}".format("Month,Year", "Sale Prediction", "Actual Value", "%"))

    for i in range(size):
        percent = 0.0
        if(predictingTarget[i] < testingTarget[i]):
            percent = predictingTarget[i] / testingTarget[i]
        else:
            percent = testingTarget[i] / predictingTarget[i]
        print("{}\t{}\t\t{}\t\t{}".format(predictingSample[i], predictingTarget[i], testingTarget[i], percent))
        total += percent
    accuracy = total/size

    print("Accuracy between the prediction and the actual sales is ", accuracy)

if __name__ == "__main__":
    main()

