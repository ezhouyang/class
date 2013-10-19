#coding:utf-8
'''
尝试利用多特征融合的手段
'''

import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

def read_file():
    print "读文件"
    f = open("train.tsv","U")
    reader = csv.reader(f,delimiter='\t')
    #用来放训练语料
    train_content = []
    train_url = []
    label = []
    
    test_content = []
    test_url = []
    answer = []

    porter = nltk.PorterStemmer()
    g = lambda x : x.isalpha or x == ' '

    a = 0
    print "开始读文件"
    for row in reader :
        if a == 0:
            a = a + 1
        else:
            row[2] = filter(g,row[2])
            raw_con = row[2].split()
            raw_con = [porter.stem(t) for t in raw_con]
            row[2] = ' '.join(raw_con)

            if a %4 != 0:
                label.append(int(row[len(row)-1]))
                train_content.append(row[2])
                train_url.append(row[0])
            else:
                answer.append(int(row[len(row)-1]))
                test_content.append(row[2])
                test_url.append(row[0])
            a = a + 1
    f.close()
    print "读文件结束"
    return train_content,train_url,label,\
        test_content,test_url,answer
    
if __name__ == "__main__":
    train_content,train_url,label,test_content,test_url,answer = read_file()
    vectorizer = TfidfVectorizer(max_features=None,min_df=2,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    vectorizer1 = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')
    
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
    answer = np.array(answer)

    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=1.0,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)

    clf1 = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=1.0,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)

    print "feature selection"
    selector = RFE(clf)
    x_content = selector.fit_transform(x_content,label)
    t_content = selector.transform(t_content)
    print "x content shape",x_content.shape
    print "t content shape",t_content.shape

    print "训练content lr"
    clf.fit(x_content,label)
    pred0 = clf.predict_proba(t_content)[:,1]
    print "content roc score",roc_auc_score(answer,pred0)

    print "训练url lr"
    clf1.fit(x_url,label)
    pred1 = clf1.predict_proba(t_url)[:,1]
    print "content roc score",roc_auc_score(answer,pred1)

    weight = 2.0

    pred = pred0*weight + pred1 
    pred = 1.0*pred/(weight+1.0)
    print "简单平均 roc score",roc_auc_score(answer,pred)
    
    
    
