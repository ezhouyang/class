# -*- coding: utf-8 -*-

from  read_tsv import *
import sklearn
import numpy as np
from sklearn import svm

if __name__ == "__main__":

    train,label = read_train("train.tsv")
    test,url_label = read_test("test.tsv")
            
    #下面的傻逼归一化程序是我自己写的，非常的难为情
    
    x = np.array(train)
    t = np.array(test)
    
    print "开始训练"
    clf = svm.SVC()
    clf.fit(x,label)
    
    print "开始预测"
    answer = clf.predict(t)

    f = open("answer.csv","w")
    f.write('urlid,label\n')

    for i in xrange(len(test)):
        f.write("%s,%s\n"%(url_label[i],int(answer[i])))
    
