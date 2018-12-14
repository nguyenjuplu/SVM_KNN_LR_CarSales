Please ensure you have the proper libraries downloaded by following installation instructions on their sites:
    https://scikit-learn.org/stable/install.html
    https://matplotlib.org/users/installing.html
    
For running the "prediction" programs for a model:
    1. Download DemandData.txt
    2. Run the program with DemandData.txt
    3. Input the time span for which will be used as the training data, in the form "StartingMonth StartYear EndMonth EndYear"
        Such input would look like "1 7 12 17" will use data from January 2007 until December 2017 for training
        The data spans from January 2007 until October 2018, so you cannot specify beyond or before that
    4. Input how many months from the end of the training you would like to predict
        You cannot predict beyond October 2018, as that is the final data point to give an accuracy
        For example: If you entered "1 7 12 17", the maximum value for months ahead would be 10
        
For running the "k-Fold" programs for a model:
    1. Download DemandData.txt
    2. Run the program with DemandData.txt
        