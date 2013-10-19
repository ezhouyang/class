#coding:utf-8
'''
利用多特征融合
'''
import csv
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

def read_file():
    print "读文件"
    f = open("train.tsv","U")
    reader = csv.reader(f,delimiter='\t')

    train_content = []
    train_url = []
    label = []
    
    test_content = []
    test_url = []
    urlid = []

    porter = nltk.PorterStemmer()
    g = lambda x : x.isalpha or x == ' '

    a = 0
    print "读训练文件"
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            row[2] = filter(g,row[2])
            raw_con = row[2].split()
            raw_con = [porter.stem(t) for t in raw_con]
            row[2] = ' '.join(raw_con)

            label.append(int(row[len(row)-1]))
            train_content.append(row[2])
            train_url.append(row[0])
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
            test_content.append(row[2])
            test_url.append(row[0])
    f.close()

    return train_content,train_url,label,\
        test_content,test_url,urlid

if __name__ == "__main__":
    train_content,train_url,label,test_content,test_url,urlid = read_file()
    vectorizer = TfidfVectorizer(max_features=None,min_df=3,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    vectorizer1 = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,3),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')

    print "构建tf-idf矩阵"
    print "构建内容矩阵"
    length_train = len(train_content)
    x_all = train_content+test_content
    x_all = vectorizer.fit_transform(x_all)
    x_content = x_all[:length_train]
    t_content = x_all[length_train:]

    print "构建url矩阵"
    length_train = len(train_content)
    x_all = train_url+test_url
    x_all = vectorizer1.fit_transform(x_all)
    x_url = x_all[:length_train]
    t_url = x_all[length_train:]

    print "x content shape",x_content.shape
    print "t content shape",t_content.shape
    print "x url shape",x_url.shape
    print "t url shape",t_url.shape
    label = np.array(label)

    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=1.0,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)

    clf1 = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=1.0,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)

    clf2 = SGDClassifier(loss="log", penalty="l2",alpha=0.0001,fit_intercept=False)#sgd 的训练结果也不错

    clf3 = SVC(C=1.0,gamma=0.3,probability=True)

    clf4 = SVC(kernel='sigmoid',degree=9,gamma=0.3,probability=True)

    print "训练content lr"
    clf.fit(x_content,label)
    pred0 = clf.predict_proba(t_content)[:,1]

    print "训练content sgd"
    clf2.fit(x_content,label)
    pred2 = clf2.predict_proba(t_content)[:,1]

    print "训练content svm rbf"
    clf3.fit(x_content,label)
    pred3 = clf3.predict_proba(t_content)[:,1]

    print "训练content svm sigmoid"
    clf4.fit(x_content,label)
    pred4 = clf4.predict_proba(t_content)[:,1]


    print "训练url lr"
    clf1.fit(x_url,label)
    pred1 = clf1.predict_proba(t_url)[:,1]

    weight = 2.0

    pred = pred0*weight + (weight) *pred2 +weight * pred3 \
        + weight*pred4+ pred1 
    pred = 1.0*pred/(4*weight+1.0)

    f = open("combine_answer.csv","w")
    f.write('urlid,label\n')

    for i in xrange(len(test_content)):
        f.write("%s,%s\n"%(urlid[i],pred[i]))
