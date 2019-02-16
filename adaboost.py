import csv
import numpy as np
from numpy import *

## Calculate misro and macro F1 scores
def f1_score_(y_true, y_pred):
    microF1 = []
    for instance in set(y_true).union(set(y_pred)):
        #print 'Digit=',instance

        # Calculate TP, TN, FP, and FN
        ind_orig = [k for k in range(len(y_true)) if y_true[k]==instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]==instance]
        TP = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'TP:', ind_orig, ind_pred, TP # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]!=instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]!=instance]
        TN = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'TN:', ind_orig, ind_pred, TN # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]!=instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]==instance]
        FP = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'FP:', ind_orig, ind_pred, FP # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]==instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]!=instance]
        FN = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'FN:', ind_orig, ind_pred, FN # DEBUG

        # Calculate micro F1 score
        if TP+FP+FN==0:
            F1 = 0.0
            print 'There is no instance of class',digit,'in true or predicted labels.'
            print 'Thus, micro F1 for class',digit,'is not included to calculate macro F1.'
        else:
            F1 = 2*float(TP)/float(2*TP+FP+FN)
            microF1.append(F1)
        #print 'micro F1:',F1

    macroF1 = np.mean(microF1)
    #print '\nmicro F1 score for each class:',microF1    
    #print 'macro F1 score:',macroF1

    return macroF1, microF1

# Load dataset and split to 4 folds
def splitDataset_4fold(fileDir):
    # Read csv and split train/test set.
    with open(fileDir, 'rb') as csvfile:
        f = csv.reader(csvfile, delimiter=';', quotechar='|')

        # Get headers of the data
        headers = f.next()

        # Get dataset
        allData = np.array([])
        for row in f:
            # Get attributes
            attr = [float(i) for i in row]
            #print len(row)

            # Reshape attributes array to use numpy array
            attrs = copy(np.asarray(attr))
            attrs = attrs.reshape(1,12)

            # Create an array which contains all attributes
            if allData.shape[0] == 0:
                allData = copy(attrs)
            else:
                allData = copy(np.vstack((allData, attrs)))

    # Shuffle the dataset
    random.seed(0)
    random.shuffle(allData)
    #print 'Dataset has a shape', allData.shape

    ## Split the whold dataset into 4 folds
    nSamples = allData.shape[0]
    nFold = 4
    nSamplesOneFold = int(nSamples/nFold)  # n = 1224
    #print 'Number of samples in each fold is', nSamplesOneFold

    fold1 = allData[                 :1*nSamplesOneFold, ]
    fold2 = allData[1*nSamplesOneFold:2*nSamplesOneFold, ]
    fold3 = allData[2*nSamplesOneFold:3*nSamplesOneFold, ]
    fold4 = allData[3*nSamplesOneFold:4*nSamplesOneFold, ]
    #print 'Shapes of fold 1, 2, 3, 4 are ',fold1.shape,fold2.shape,fold3.shape,fold4.shape

    test1 = copy(fold1)
    train1 = copy(np.vstack((fold2, fold3, fold4)))
    test2 = copy(fold2)
    train2 = copy(np.vstack((fold1, fold3, fold4)))
    test3 = copy(fold3)
    train3 = copy(np.vstack((fold1, fold2, fold4)))
    test4 = copy(fold4)
    train4 = copy(np.vstack((fold1, fold2, fold3)))    

    return train1,test1,train2,test2,train3,test3,train4,test4

# Divide dataset into data (nExamples x nAtt) and label (nExamples x 0)
def divide_data_and_label(dataset):
    data = dataset[:,:-1]
    label = dataset[:,-1]
    #print data.shape, label.shape
    return data, label

# Make labels binary so we can do one-vs-all
def makeLabelBinary(classLabels, class_):
    classLabelsBin = 1*(mat(classLabels) == class_) + -1*(mat(classLabels) != class_)
    #print classLabelsBin
    return classLabelsBin

# Binary classification using stump
def decisionStump_classify(dataMat, dimen, threshVal, ineq):
    retArr = ones((dataMat.shape[0], 1))
    if ineq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1.0
    return retArr

# Build decision stump
def build_decisionStump(dataArr, classLabels, D):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = dataMat.shape
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = min(dataMat[:, i]); rangeMax = max(dataMat[:, i])
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predicatedVal = decisionStump_classify(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #for the row that predicatedVal == labelMat, errArr[row] = 0
                errArr[predicatedVal == labelMat] = 0
                weightedError = D.T * errArr
                #print 'split: dim %d, thesh %.2f, ineqal: %s, \
                #weighted error:%.3f' %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predicatedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    
    return bestStump, minError, bestClassEst   

# AdaBoost using Decision stumps (depth-1 tree) as weak learner
def adaBoostDS(dataArr, classLabels, nClass = 10, numIter = 30):
    bestStumpArr = []
    nExample = dataArr.shape[0]
    D = mat(ones((nExample,1))/nExample)
    aggClassEst = mat(zeros((nExample,1)))
    
    for i in range(numIter):
        #print 'iter', i
        bestStump, error, bestClassEst = build_decisionStump(dataArr, classLabels, D)
        #print 'D:', D.T
        
        #print 'error:', error
        if error > (1-1/nClass):
            break
            
        #alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)) + log(nClass-1))
        #print 'alpha:', alpha
        bestStump['alpha'] = alpha
        bestStumpArr.append(bestStump)
        #print 'ClassEst:', bestClassEst.T
        
        #multiply(): element-wise product. class real result X estimation
        expon = multiply(-1 * alpha * mat(classLabels).T, bestClassEst) #####
        #exp(expon): calculate exp for each element in mat expon
        D = multiply(D, exp(expon)) / sum(D)
        
        #aggClassEst is float mat.
        aggClassEst += alpha * bestClassEst
        #print 'aggClassEst:', aggClassEst
        
        #aggClassEst is float mat, use its sign to compare with mat classLabels
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((nExample,1)))
        errorRate = sum(aggError)/nExample
        #print 'total error:', errorRate
        
        if errorRate == 0.0:
            break
        
    return bestStumpArr

# Test the learned AdaBoost model
def test_adaboost(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = decisionStump_classify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print aggClassEst
    #return sign(aggClassEst)
    #print aggClassEst
    return aggClassEst


### Main script
## 1. Load data and prepare dataset for 4-fold cross-validation
filename = 'winequality-white.csv'
train1,test1,train2,test2,train3,test3,train4,test4 = splitDataset_4fold(filename)
x_train, y_train = divide_data_and_label(train4)


## 2. Train 10 adaBoostDS classifiers
classes = range(0,10)
aggClassEstAll = []
bestStumpArrAll = []
for class_ in classes:
    #print 'AdaBoost Classifier for Class =',class_
    #print '-------------1)DATAPREP-------------'
    #dataArr, labels = loadSimpleData(class_)
    labels = makeLabelBinary(y_train, class_)
    #print '-------------2)TRAINING-------------'
    bestStumpArr = adaBoostDS(x_train, labels, nClass = 10, numIter = 1)
    #print '----------3)CLASSIFICATION----------'
    aggClassEst = test_adaboost(x_train, bestStumpArr)
    aggClassEstAll.append(aggClassEst)
    bestStumpArrAll = np.hstack((bestStumpArrAll, bestStumpArr))
      
# Find the class label that gives best score for each example in Training set
#print '---------Evaluate All Classes---------'
aggClassEstAll = np.squeeze(np.asarray(aggClassEstAll))
#print 'score matrix shape:',aggClassEstAll.shape
#print 'score matrix (row=class, col=data):\n',aggClassEstAll

ypred_idx = np.argmax(aggClassEstAll, axis=0)
#print 'predicted labels:', ypred_idx.shape, ypred_idx
#print 'true labels:', y_train.shape, y_train
# Summary of performance
#print 'accuracy =',double(sum(ypred_idx==y_train))/double(len(y_train)),',', sum(ypred_idx==y_train),'out of',len(y_train),'is correctly predicted.'


## 3. Test 
x_test, y_test = divide_data_and_label(test4)

classes = range(0,10)
aggClassEstAll = []
for class_ in classes:
    #print class_
    aggClassEst = test_adaboost(x_test, [bestStumpArrAll[class_]])
    aggClassEstAll.append(aggClassEst)
    
# Find the class label that gives best score for each example in Test set
#print '---------Evaluate All Classes---------'
aggClassEstAll = np.squeeze(np.asarray(aggClassEstAll))
#print 'score matrix shape:',aggClassEstAll.shape
#print 'score matrix (row=class, col=data):\n',aggClassEstAll

ypred_idx = np.argmax(aggClassEstAll, axis=0)
#print 'predicted labels:', ypred_idx.shape, ypred_idx
#print 'true labels:', y_test.shape, y_test
# Summary of performance
#print 'accuracy =',double(sum(ypred_idx==y_test))/double(len(y_test)),',',sum(ypred_idx==y_test),'out of',len(y_test),'is correctly predicted.'

macroF1, microF1 = f1_score_(y_test, ypred_idx)

print 'Using 25% of the dataset for Testing'
print 'Testset Macro F1:',macroF1
print 'Testset Micro F1 for class=3,4,5,6,7,8,9:',microF1



