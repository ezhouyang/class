#coding:utf-8

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
import csv
from scipy import sparse

def read_file():
    print "read_train"
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
            #选择第二项作为训练文章
            #text_train.append(row[2]+" "+row[0]+" "+row[3])
            text_train.append(row[2]+" "+row[0])
            #处理一下其他feature
            row[17]=0
            row[20]=0
            row[21]=0
            row[10]=0
            row[19] = float(row[19])/100
            row[22] = float(row[22])/200
            row[23] = float(row[23])/20
            extra_train.append([float(i)/100.0 for i in row[5:len(row)-1]])

    f.close()
    print "read test"
    f = open("test.tsv","U")
    reader = csv.reader(f,delimiter='\t')
    
    a = 0
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            urlid.append(row[1])
            #text_test.append(row[2]+" "+row[0]+" "+row[3])
            text_test.append(row[2]+" "+row[0])
            row[17]=0
            row[20]=0
            row[21]=0
            row[10]=0
            row[19] = float(row[19])/100
            row[22] = float(row[22])/200
            row[23] = float(row[23])/20
            extra_test.append([float(i)/100.0 for i in row[5:len(row)]])
    
    return text_train,label,text_test,urlid,extra_train,extra_test

def remain(answer):
    """
    
    Arguments:
    - `answer`:
    """
    
    for i in range(len(answer)):
        if answer[i] > 0.9725:
            answer[i] = 1.0
    return answer

if __name__ == "__main__":
    train,label,test,urlid,extra_train1,extra_test1 = read_file()
    print "train length",len(train)
    print "test length",len(test)
    vectorizer = TfidfVectorizer(sublinear_tf=True,min_df = 3,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',use_idf=1,analyzer='word',strip_accents='unicode')
    print "transform train to tf matrix"
    print "transform test to tf matrix"
    length_train = len(train)
    x_all = train + test
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    extra_train,extra_test = [],[]

    print "读topic"
    f1 = open("topic_train.txt")
    for line in f1.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        extra_train.append(sp)

    f2 = open("topic_test.txt")
    for line in f2.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        extra_test.append(sp)

    extra_train = np.array(extra_train)
    extra_test  = np.array(extra_test)

    print "合并特征"
    x = sparse.hstack((x,extra_train)).tocsr()
    t = sparse.hstack((t,extra_test)).tocsr()
    #x = sparse.hstack((x,extra_train1)).tocsr()
    #t = sparse.hstack((t,extra_test1)).tocsr()

    label = np.array(label)
    #clf = svm.SVC(kernel='sigmoid',degree=9,gamma=10)
    #clf = svm.SVC(degree=9,gamma=0.001)
    #clf = KNeighborsClassifier(n_neighbors=1)
    #clf = LogisticRegression(penalty='l2',C=300,tol=1e-6)
    #clf = SGDClassifier(loss="log",n_iter=300, penalty="l2",alpha=0.0003)
    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=2,tol=1e-9,class_weight=None, random_state=None, intercept_scaling=1.0)
    print "交叉验证"
    print np.mean(cross_validation.cross_val_score(clf,x,label,cv=20,scoring='roc_auc'))
    clf.fit(x,label)
    #验一下自己的结果
    print "训练自己",clf.score(x,label)
    answer =  clf.predict_proba(t)[:,1]
    answer = remain(answer)
    
    f = open("hand_answer.csv","w")
    f.write('urlid,label\n')

    for i in xrange(len(test)):
        f.write("%s,%s\n"%(urlid[i],answer[i]))
    
    

