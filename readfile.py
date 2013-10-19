#coding: utf-8

import csv

def read_file():
    
    print "读文件"
    f = open("train.tsv","U")
    reader = csv.reader(f,delimiter='\t')

    t = open("total.txt","w")

    train = []
    test = []
    
    a = 0
    print "读训练文件"
    for row in reader :
        if a == 0:
            a = a + 1
        else:
            train.append(1)
            t.write(row[2]+" "+row[0]+"\n")

    f.close()

    a = 0
    f = open("test.tsv","U")
    reader = csv.reader(f,delimiter='\t')
    
    print "读测试文件"
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            test.append(2)
            t.write(row[2]+" "+row[0]+"\n")

    print "train",len(train)
    print "test",len(test)

if __name__ == "__main__":
    read_file()
