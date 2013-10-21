#coding: utf-8
'''
这个文件的作用是将输入的doc-topic文件读成两个文件，并将topic还原
'''

if __name__ == "__main__":
    f = open("class.topic")

    train = []
    test = []

    total = []

    for i in f.readlines():
        total.append(i)
        
    total = total[1:]

    train_length = 7395
    train = total[:train_length]
    test = total[train_length:]

    print "train length",len(train)
    print "test length",len(test)
    
    t1 = open("topic_train.txt","w")
    t2 = open("topic_test.txt","w")

    topic_num = 70
    topic_num1 = topic_num*2
    
    for line in train:
        sp = line.split('\t')
        sp = sp[2:len(sp)-1]
        temp = [0 for i in range(topic_num1)]

        for i in range(topic_num):
            ind = int(sp[i*2])
            temp[ind] = float(sp[i*2+1])
        st = [str(j) for j in temp]
        st = ' '.join(st)
        t1.write(st+'\n')

    for line in test:
        sp = line.split('\t')
        sp = sp[2:len(sp)-1]
        temp = [0 for i in range(topic_num1)]

        for i in range(topic_num):
            ind = int(sp[i*2])
            temp[ind] = float(sp[i*2+1])
        st = [str(j) for j in temp]
        st = ' '.join(st)
        t2.write(st+'\n')


    
