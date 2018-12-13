import numpy as np
import scipy
from scipy import stats

f = open('/home/cloudera/cmpe139hw1/train.dat', "r")
userlist = []
itemlist = []
trainData = []
for row in f:
    r = row.split('\t')
    userlist.append(int(r[0]))
    itemlist.append(int(r[1]))
    trainData.append([int(r[0]), int(r[1]), int(r[2])])

f.close()
users = max(userlist) + 1
items = max(itemlist) + 1

utilMat = np.zeros((users, items))
for entry in trainData:
    utilMat[entry[0]][entry[1]] = entry[2]

psim = np.zeros((items, items))
for item1 in range(items):
    for item2 in range(items):
        if np.any(utilMat[:, item1]) and np.any(utilMat[:, item2]):
             print str(item1) + ", " + str(item2)
             psim[item1][item2] = scipy.stats.pearsonr(utilMat[:, item1], utilMat[:, item2])[0]

f = open('/home/cloudera/cmpe139hw1/test.dat', "r")
testSet = []
for row in f:
    r = row.split('\t')
    testSet.append([int(r[0]), int(r[1])])

f.close()
w = open('/home/cloudera/cmpe139hw1/format.dat', "w")
for i in range(len(testSet)):
    uid = testSet[i][0]
    iid = testSet[i][1]
    itemSimVector = psim[iid]
    userRatingVector = utilMat[uid]
    simSum = np.sum(psim)
    pred = np.dot(itemSimVector, userRatingVector)/simSum
    print pred
    if pred > 5:
        pred = 5
    elif pred < 0:
        pred = 0
    w.write(str(pred) + "\n")

w.close()
