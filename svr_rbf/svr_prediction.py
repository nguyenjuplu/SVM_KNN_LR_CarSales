import sys

def printResult(predictingSample, predictingTarget, actualTarget):
    """
    print out the sale prediction and its corresponding year
    """
    total = 0.0
    size = len(predictingSample)
    print("Sale prediction in {} months: ".format(size))
    print("{}\t\t{}".format("Mth,Yr", "Sale prediction"))
    for i in range(size):
        print("{}\t\t{}".format(predictingSample[i], predictingTarget[i]))


def svrPrediction(sampleList, targetList, predictingSampleList):
    """
        The svr_rbf prediction function that takes in 2D sample list and 1D target list, and return the 1D predicting list.
    """
    from sklearn import svm
    X = sampleList
    y = targetList
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma='scale')
    y_rbf = svr_rbf.fit(X, y).predict(predictingSampleList)
    y_rbf = [int(round(x)) for x in y_rbf]
    #use for graphing to show trace of training model
    y_modelTraining = svr_rbf.fit(X, y).predict(sampleList)
    return y_rbf, y_modelTraining

def plotGraph(trainingSample, trainingTarget, predictingSample, predictingTarget, trainingModelTarget):
    """
    Plot the graph in x-y coordinate.
    X-cor:
        convert trainingSample and predictingSample from 2D to 1D by making it a decimal presenting the month and year
        ex: [1,8] = 12(8-7) + 1 = 13
    Y-cor:
        predictingTarget - input of sale data
        predictingTarget - predicting sale
        trainingModelTarget - tracing of the training model
    """

    import matplotlib.pyplot as plt
    lw = 2
    startMth = trainingSample[0][0]
    xSample = [i+startMth for i in range(len(trainingSample))]
    xPredict = [i+1 for i in range(xSample[-1], xSample[-1] + len(predictingSample))]
    plt.scatter(xSample, trainingTarget, color='darkorange', label='Data Sample')
    plt.plot(xSample, trainingModelTarget, color='c', lw=lw, label='RBF model')
    plt.scatter(xPredict, predictingTarget, color='navy', lw=lw, label='RBF prediction')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

def splitDataSet(sampleLst, targetLst, frmMth, frmYr, toMth, toYr, nxtMth, testing):
    """
   Given from what month of the year to what month the year, and the number of predicting month ahead of time,
   the funtion return 2D list of the training sample and the predicting sample, and the 1D of the training traget and the real value of the predicting target
    """

    numSample = (toYr-frmYr+1)*12 - (frmMth-1) - (12 - toMth)
    frmIndex = (frmYr - 7) * 12 + (frmMth-1)
    toIndex = frmIndex + numSample

    trningSample = sampleLst[frmIndex:toIndex]
    trningTarget = targetLst[frmIndex:toIndex]

    mth = toMth
    yr  = toYr
    predictSample = []
    for i in range(nxtMth):
        mth += 1
        if(mth == 13):
            mth = 1
            yr += 1
        predictSample.append([mth, yr])

    if testing == "test-yes": 
        testTarget = targetLst[toIndex:toIndex+nxtMth]
        return  trningSample, trningTarget, predictSample, testTarget
    else:
        return trningSample, trningTarget, predictSample
    

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

def CheckInput(argvs):
    """
    check to see whether the user input the text file.
    """
    if len(sys.argv) != 2:
        print("Input dataset textfile is not found.")
        return False
    return True

def main():
    #checking input file
    if(not CheckInput(sys.argv)):
        sys.exit(0)
    inputFile =  sys.argv[1] 

    #dataSet is in term of month and year, target is the corresponding sale 
    dataSet, target = extractData(inputFile)

    #getting user input for model training and prediction 
    #variables: frmMth, frmYr, toMth, toYr, nextMths
    print("Dataset start from 01_07 to 10_18")
    print("Enter a desired length for model training serperated by space:")
    user_input = input("From month year to month year:")
    user_input = user_input.split()
    if(len(user_input) != 4):
        print("Incorrect input!!!")
        sys.exit(0)
    frmMth, frmYr, toMth, toYr = [int(x) for x in user_input]
    nextMths = input("How many month ahead you want to predict? ")
    nextMths = int(nextMths)

    #splitting Data set to training sample, training target, prediction sample, and optional testing target 
    trainingSample, trainingTarget, predictingSample, testingTarget = splitDataSet(dataSet, target, frmMth, frmYr, toMth, toYr, nextMths, "test-yes")
    #trainingModelTarget is only for the graph to show the trace of the training model before the prediction
    predictingTarget, trainingModelTarget = svrPrediction(trainingSample, trainingTarget, predictingSample)

    # comparing the predicting sale and the actual sale 
    # print the predicting result and accuracy
    printResult(predictingSample, predictingTarget, testingTarget)

    #plotting the training graph, the prediction graph, and the trace of the training model
    plotGraph(trainingSample, trainingTarget, predictingSample,predictingTarget , trainingModelTarget)


if __name__ == "__main__":
    main()