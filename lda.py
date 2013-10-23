#coding: utf-8

import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def read_file():
    print "读训练文件"
    f = open("train.tsv","U")
    reader = csv.reader(f,delimiter='\t')
    #用来存放训练集的文章
    text_train = []
    #用来存放训练集的标签
    label = []
    #用来存放测试集的文章
    text_test = []
    #用来存放urlid
    urlid = []

    extra_train = []
    extra_test = []

    g = lambda x : x.isalpha or x == ' '
    
    a = 0
    print "read train file begin"
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            #标签为最后一项
            label.append(int(row[len(row)-1]))

    f.close()

    print "读测试文件"
    f = open("test.tsv","U")
    reader = csv.reader(f,delimiter='\t')
    
    a = 0
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            urlid.append(row[1])


    return label,urlid

if __name__ == "__main__":
    label,urlid = read_file()
    
    train,test = [],[]

    print "读topic"
    f1 = open("topic_train6.txt")
    for line in f1.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        train.append(sp)

    f2 = open("topic_test6.txt")
    for line in f2.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        test.append(sp)

    x = np.array(train)
    t  = np.array(test)

    #x = np.hstack((x,extra_train))
    #t = np.hstack((t,extra_test))

    print x.shape

    label = np.array(label)

    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=True,C=20,tol=1e-9,class_weight=None, random_state=None, intercept_scaling=1.0)
    #clf = AdaBoostClassifier(n_estimators=100)
    #clf = SGDClassifier(loss="log",n_iter=300, penalty="l2",alpha=0.00005,fit_intercept=Tr1ue)#sgd 的训练结果也不错
    #clf = svm.SVC(C=1,degree=9,gamma=10,probability=True)
    #clf = RandomForestClassifier(n_estimators=2000, criterion ="gini",max_depth=20,min_samples_split=1, random_state=0,n_jobs=3)
    #clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,max_depth=1, random_state=0)
    print "cross validation"
    print np.mean(cross_validation.cross_val_score(clf,x,label,cv=20,scoring='roc_auc'))

    clf.fit(x,label)
    #验一下自己的结果
    print "训练自己",clf.score(x,label)
    answer =  clf.predict_proba(t)[:,1]
    
    f = open("hand_answer.csv","w")
    f.write('urlid,label\n')

    for i in xrange(len(test)):
        f.write("%s,%s\n"%(urlid[i],answer[i]))
    
    
