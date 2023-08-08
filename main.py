import random
import pandas as pd
import time
from decisionTree  import buildDecisionTree,decisionTreePredictions , calculateAccuracy,trainTestSplit,singlePredictions
from explainprediction import explain

dataFrame = pd.read_csv("breast_cancer.csv")
dataFrame = dataFrame.drop("id", axis = 1)
dataFrame = dataFrame[dataFrame.columns.tolist()[1: ] + dataFrame.columns.tolist()[0: 1]]
dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize = 0.25)

print("Decision Tree - Breast Cancer Dataset")

i = 1
accuracyTrain = 0
while accuracyTrain < 100:
    startTime = time.time()
    decisionTree = buildDecisionTree(dataFrameTrain, maxDepth = i)
    buildingTime = time.time() - startTime
    decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
    accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
    accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("maxDepth = {}: ".format(i), end = "")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
    i += 1

cols = dataFrameTest.columns
dataFramesingleTest = dataFrameTest.iloc[16,:]
first_row = dataFramesingleTest.values
first_row = first_row.reshape(1,len(cols))

check = pd.DataFrame(first_row,columns = cols)


decisionTreeTestResults = singlePredictions(check, decisionTree)

explain(decisionTreeTestResults)
