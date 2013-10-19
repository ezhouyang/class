#coding:utf-8

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import nltk
from scipy import sparse

import csv

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
    answer = []
    #用来存放额外的feature
    extra_train = []
    extra_test = []

    porter = nltk.PorterStemmer()
    g = lambda x : x.isalpha or x == ' '
    
    a = 0
    print "read train file begin"
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            row[17]=0
            row[20]=0
            row[21]=0
            row[10]=0
            row[19] = float(row[19])/100
            row[22] = float(row[22])/200
            row[23] = float(row[23])/20
            #处理一下字符串
            #row[2] = filter(g,row[2])
            #sp = row[2].split('"')
            #if len(sp)>7:
            #    raw_con = (sp[3]+sp[7]).split()
            #else:
            #    raw_con = (sp[3]).split()
            #print raw_con
            #raw_con = [porter.stem(t) for t in raw_con]
            #if len(row[2]) > 30:
            #    row[2] = row[2][:30]
            #row[2] = ' '.join(raw_con)
            #row[8] = 0
            if a % 3 != 0:
            #标签为最后一项
                label.append(int(row[len(row)-1]))
            #选择第二项作为训练文章
                text_train.append(row[2]+" "+row[0])
                extra_train.append([float(i) for i in row[5:len(row)-1]])
            else:
                answer.append(int(row[len(row)-1]))
                text_test.append(row[2]+" "+row[0])
                extra_test.append([float(i) for i in row[5:len(row)-1]])
            a = a + 1

    f.close()
    
    
    return text_train,label,text_test,answer,extra_train,extra_test

if __name__ == "__main__":
    train,label,test,ans,extra_train,extra_test = read_file()

    #vectorizer = TfidfVectorizer(max_features=None,min_df=3,max_df=0.4,sublinear_tf=False,ngram_range=(1,3),smooth_idf=True,token_pattern=r'\w{1,}',use_idf=1,analyzer='word',strip_accents='unicode')
    vectorizer = TfidfVectorizer(max_features=None,min_df=3,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    #vectorizer = CountVectorizer(min_df=0.02,max_df=0.1,stop_words='english')
    print "transform train to tf matrix"
    #*********semi data!
    #train.extend(test)
    #label.extend([-1 for s in xrange(len(test))])
    #****************
    length_train = len(train)
    x_all = train + test
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]
    #x = sparse.hstack((x,extra_train)).tocsr()
    #t = sparse.hstack((t,extra_test)).tocsr()
    
    #x = vectorizer.fit_transform(train)
    print x
    print "transform test to tf matrix"
    #t = vectorizer.transform(test)

    print x.toarray
    
    
    print "feature selection"
    #clf = LinearSVC(C=50, penalty="l1", dual=True)
    #svc = SVC(kernel="linear")
    #rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(label, 2000),scoring='accuracy')
    #rfecv.fit(x,label)
    #rfecv.transform(x)
    #rfecv.transform(t)
    #print "x",x.shape
    #print "t",t.shape

    #print "add features"
    #new_x = []
    #new_t = []
    #for j in xrange(len(x)):
    #    lj = list(x[j])
    #    lj.extend(extra_train[j])
    #    new_x.append(lj)
    #clf = LogisticRegression(penalty='l1',C=100,tol=1e-3)
    #x = clf.fit_transform(x,label)
    #t = clf.transform(t)
    #print "new x features",x.shape
    #print "new t features",t.shape

    #print "svd"
    #pca = PCA(n_components=200)
    #pca.fit(x.toarray())
    #lsa = TruncatedSVD(n_components=400)
    #x = lsa.fit_transform(x)
    #t = lsa.transform(t)


        
    #for j in xrange(len(t)):
    #    lj = list(t[j])
    #    lj.extend(extra_train[j])
    #    new_t.append(lj)

    #x = np.array(new_x)
    #t = np.array(new_t)

    label = np.array(label)
    ans = np.array(ans)

    print "training ..."
    #clf = svm.SVC(kernel='sigmoid',degree=9,gamma=0.1)
    clf = svm.SVC(C=5,degree=9,gamma=0.3,probability=True)
    #clf = svm.SVC(kernel='poly',degree=9,gamma=0.008)
    #clf = KNeighborsClassifier(n_neighbors=5)#10是最高的
    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=2,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)
    #pred0 = clf.predict_proba(t)[:,1]
    #clf = LogisticRegression(penalty='l1',C=0.1,tol=1e-10)
    #clf = AdaBoostClassifier(n_estimators=20)
    #clf = BernoulliNB()
    #clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1, random_state=0)
    #clf = SGDClassifier(loss="log",n_iter=300, penalty="l2",alpha=0.0003,fit_intercept=True)#sgd 的训练结果也不错
    print "交叉训练"
    print "cross validation",np.mean(cross_validation.cross_val_score(clf,x,label,cv=5,scoring='roc_auc'))
    clf.fit(x,label)
    #print "训练自己",clf.score(x,label)
    #print "训练样本",clf.score(t,ans)
    #pred1 = clf.predict_proba(t)[:,1]
    #pred = (pred0+pred1*2)/3.0
    #print pred
    pred = clf.predict_proba(t)[:,1]
    #print "roc auc score 0 ",roc_auc_score(ans,pred0)
    #print "roc auc score 1",roc_auc_score(ans,pred1)
    print "roc auc score final",roc_auc_score(ans,pred)
    

    
    
    
    

