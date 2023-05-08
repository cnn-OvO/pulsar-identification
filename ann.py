from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data import DataLoader
import time 
import sys
import pickle
import numpy as np
from coatnet import CoAtNet
from utils import outputanalysis
from train_svm import readtxt
from traingpps import MYDataset
import pandas as pd

def profdm_model(trainfile, testfile, epoch = 200):
    train_pfd, X_train, Y_train = readtxt(trainfile, num_works=40)
    clf = MLPClassifier(hidden_layer_sizes=(528, 1056, 2112, 1056, 352, 2), random_state=1, verbose=True,
                        max_iter=epoch, learning_rate='adaptive',learning_rate_init=0.001,solver='sgd',tol=0.001)
    clf.fit(X_train, Y_train)
    Y_dec = clf.predict_proba(X_train)[:,1]
    rec, fpr, acc, f1 = outputanalysis(Y_dec, Y_train, threshold = 0.1)
    print("Recall of traindataset = %.4f"%(rec))
    print("False Positive Rate of traindataset = %.4f"%(fpr))
    print("Accuracy of traindataset = %.4f"%(acc))
    print("F1 score of traindataset= %.4f"%(f1))
    test_pfd, X_test, Y_test = readtxt(testfile, num_works= 40)
    Y_dec = clf.predict_proba(X_test)[:,1]
    rec, fpr, acc, f1 = outputanalysis(Y_dec, Y_test, threshold = 0.1)
    print("Recall of testdataset = %.4f"%(rec))
    print("False Positive Rate of testdataset = %.4f"%(fpr))
    print("Accuracy of testdataset = %.4f"%(acc))
    print("F1 score of testdataset= %.4f"%(f1))
    with open("trained_model/ann_profdm_3000.pkl", 'wb') as f:
        pickle.dump(clf, f)

def prof_model(trainfile, testfile, epoch = 200):
    trainpfd, X_train, Y_train = readtxt(trainfile, num_works=60, data_type='dm')
    clf = MLPClassifier(hidden_layer_sizes=(528, 1056, 352, 2),random_state=1, verbose=True,
                        max_iter=epoch, learning_rate='adaptive',learning_rate_init=0.01,solver='sgd')
    clf.fit(X_train, Y_train)
    Y_dec = clf.predict_proba(X_train)[:,1]
    acc , f1 = outputanalysis(Y_dec,Y_train)
    print("Accuracy of traindataset = %.4f"%(acc))
    print("F1 score of traindataset= %.4f"%(f1))
    test_pfd, X_test, Y_test = readtxt(testfile, num_works= 60, data_type='dm')
    Y_dec = clf.predict_proba(X_test)[:,1]
    acc , f1 = outputanalysis(Y_dec,Y_test)
    print("Accuracy of testdataset = %.4f"%(acc))
    print("F1 score of testdataset= %.4f"%(f1))


def testtwomodel(txtfile, model1, model2):
    pfd, X_train, Y_train = readtxt(txtfile, num_works=30, data_type='1Ddata')
    with open(model1,"rb") as f:
        clf = pickle.load(f)
    X1 = clf.predict_proba(X_train)
    print("get score of profdm_model.")
    num_blocks = [2, 2, 3, 5, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    channel = 3
    device = "cpu"
    model = CoAtNet((64,64),channel,num_blocks, channels, num_classes=2,block_types=['C', 'C', 'C', 'T'])
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model2,map_location=torch.device(device)))
    model.eval()
    dataset = MYDataset(txtfile,dir="", channel=3)
    dataloader = DataLoader(dataset,batch_size=128,num_workers=40,pin_memory=True)
    X2 = np.empty((0,2))
    for _, data in enumerate(dataloader,0):
        data, _ = data
        output_val = model(data)
        S = torch.nn.Softmax(dim=1)
        output_val = S(output_val)
        X2 = np.concatenate((X2,output_val.cpu().detach().numpy()),axis=0)
    print("get score of coatnet_model.")
    X = np.concatenate([X1,X2],axis=1)

    return pfd, X, Y_train


def top_model(trainfile, testfile, epoch=200, model1 = 'trained_model/profdm_ann.model', model2 = 'trained_model/ccct-22352-train3-3chs-21.model'):
    t0 = time.time()
    trainpfd, X_train, Y_train = testtwomodel(trainfile, model1, model2)
    print("get traindata in %.2f"%(time.time()-t0))
    #df1 = pd.DataFrame(trainpfd)
    #df2 = pd.DataFrame(X_train)
    #df3 = pd.DataFrame(Y_train)
    #df = pd.concat([df1,df2,df3],axis=1)
    #del df1, df2, df3
    #df.to_csv("down_layer.csv", sep='\t', header=None, index=False)
    #clf = MLPClassifier(hidden_layer_sizes=(16, 64, 32, 1), random_state=1, verbose=True,
    #                    max_iter=epoch, learning_rate='adaptive',learning_rate_init=0.001,solver='sgd')
    #clf.fit(X_train, Y_train)
    
    C = [0.01, 0.1, 1.0, 2.0]
    for c in C:
        print("C = %f"%(c))
        clf = LogisticRegression(C=c,tol=0.00001).fit(X_train, Y_train)

        Y_dec = clf.predict_proba(X_train)
        rec, fpr, acc, f1 = outputanalysis(Y_dec,Y_train,threshold = 0.5)
        print("Recall of traindataset = %.4f"%(rec))
        print("False Positive Rate of traindataset = %.4f"%(fpr))
        print("Accuracy of traindataset = %.4f"%(acc))
        print("F1 score of traindataset= %.4f"%(f1))
        t0 = time.time()
        testpfd, X_test, Y_test = testtwomodel(testfile, model1, model2)
        print("get testdata in %.2f"%(time.time()-t0))
        Y_dec = clf.predict_proba(X_test)
        rec, fpr, acc , f1 = outputanalysis(Y_dec,Y_test,threshold = 0.5)
        print("Recall of testdataset = %.4f"%(rec))
        print("False Positive Rate of testdataset = %.4f"%(fpr))
        print("Accuracy of testdataset = %.4f"%(acc))
        print("F1 score of testdataset= %.4f"%(f1))
        with open("trained_model/top_lr_1000_"+str(c)+".model", 'wb') as f:
            pickle.dump(clf, f)




if __name__ == '__main__':
    traintxt = sys.argv[1]
    testtxt = sys.argv[2]
    #top_model(traintxt, testtxt, model1="./trained_model/ann_profdm_1000.pkl", model2="./trained_model/coatnet-v5-3chs-76.model", epoch = 800)
    profdm_model(traintxt, testtxt, epoch = 800)