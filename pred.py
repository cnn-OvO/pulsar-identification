import os
import argparse
import glob
import time
import pickle
from coatnet import CoAtNet
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from utils import load_pred_data, readpfd, from_idx_read
from torchvision.transforms.functional import rotate

CurDir = os.path.split(os.path.abspath(__file__))[0]
print(CurDir)

class idx_Dataset(Dataset):
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return index

class pred_Dataset(Dataset):
    def __init__(self, pfds, channel):
        self.pfds = pfds
        self.channel = channel

    def __len__(self):
        return len(self.pfds)

    def __getitem__(self, idx):
        if self.channel == 2:
            data = torch.as_tensor(readpfd().get2Ddata(self.pfds[idx],resize_bin=64),dtype=torch.float32)
        elif self.channel == 3:
            data = torch.as_tensor(readpfd().get3Chansdata(self.pfds[idx],resize_bin=64),dtype=torch.float32)
        elif self.channel == 4:
            data = torch.as_tensor(readpfd().get2Ddata(self.pfds[idx],resize_bin=64),dtype=torch.float32)
            data2 = rotate(data,angle=90)
            data = torch.cat([data,data2],dim=0)
            del data2
        else: 
            pass
        return data


class predict(object):
    def __init__(self, testfile = None, decimal = 5):
        if testfile is None:
            self.pfds = glob.glob("*.pfd")
        else:
            with open(testfile, 'r') as f:
                self.pfds = []
                for line in f:
                    line = line.split()
                    self.pfds.append(line[0])
        self.decimal = decimal

    def writetxt(self, scores, txtname, decimal = None):
        if decimal is None:
            decimal = self.decimal
        scores = np.around(scores,decimals=decimal)
        text = "\n".join(["%s %f" %(self.pfds[i], scores[i]) for i in range(len(self.pfds))])
        with open(txtname, "w") as f:
            f.write(text)

    def prof_svm_pred(self, modelname, num_works = 50, scoretxt="prof_svm.score",decimal=None):
        X = load_pred_data(self.pfds, num_works=num_works, data_type="profile")
        with open(modelname, "rb") as f:
            clf = pickle.load(f)
        Y = clf.predict_proba(X)[:,1]
        self.writetxt(Y, scoretxt, decimal=decimal)


    def profdm_ann_test(self,modelname=CurDir+"/trained_model/profdm_ann.model", num_works=50):
        X = load_pred_data(self.pfds, num_works=num_works, data_type="1Ddata")
        with open(modelname,"rb") as f:
            clf = pickle.load(f)
        Y = clf.predict_proba(X)
        return Y
    def profdm_ann_pred(self, modelname=CurDir+"/trained_model/profdm_ann.model", scoretxt = "profdm_ann.score", num_works=50,decimal=None):
        Y = self.profdm_ann_test(modelname=modelname,num_works=num_works)[:,1]
        self.writetxt(Y, scoretxt, decimal=decimal)


    def coatnet_test(self, modelname=CurDir+"/trained_model/ccct-22352-train3-3chs-21.model", channel=3, num_works=50):
        testdataset = idx_Dataset(length=len(self.pfds))
        if len(self.pfds) > 10000:
            batch_size = 10000
        else: 
            batch_size = len(self.pfds)
        testloader = DataLoader(dataset=testdataset,batch_size=batch_size)
        #print(batch_size)
        num_blocks = [2, 2, 3, 5, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        model = CoAtNet((64,64),channel,num_blocks, channels, num_classes=2, block_types=['C', 'C', 'C', 'T'])
        device = torch.device("cpu")
        model.to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(modelname,map_location=torch.device(device)))
        print("loaded modelstate done! ")
        t0 = time.time()
        model.eval()
        outputs = np.empty((0,2))
        for _, data in enumerate(testloader,0):
            batch_pfds = [self.pfds[i] for i in data]
            data = from_idx_read(batch_pfds,num_works=num_works)
            output_val = model(data)
            #print(time.time()-t)
            S = nn.Softmax(dim=1)
            output_val = S(output_val)
            outputs = np.concatenate((outputs,output_val.cpu().detach().numpy()),axis=0)
        return outputs
    def coatnet_pred(self, modelname=CurDir+"/trained_model/ccct-22352-train3-3chs-21.model",scoretxt="coatnet_3chs.score", channel=3, num_works=50, decimal=None):
        Y = self.coatnet_test(modelname=modelname,scoretxt=scoretxt,channel=channel,num_works=num_works)[:,1]
        self.writetxt(Y,scoretxt,decimal=decimal)


    def topmodel_pred(self,topmodelname,model1,model2,scoretxt="result_score_ACLR.txt",channel=3,num_works=50,decimal=None):
        t0 = time.time()
        X1 = self.profdm_ann_test(modelname=model1,num_works=num_works)
        print("predict profdm done in %.2f"%(time.time()-t0))
        t1 = time.time()
        X2 = self.coatnet_test(modelname=model2,channel=channel,num_works=num_works)
        print("predict 2-D features done in %.2f"%(time.time()-t1))
        X = np.concatenate([X1,X2],axis=1)
        with open(topmodelname,"rb") as f:
            clf = pickle.load(f)
        Y = clf.predict_proba(X)[:,1]
        self.writetxt(Y,scoretxt,decimal=decimal)
        print("all done in %.2f"%(time.time()-t0))


def get_parse():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('-f',type=str,help="The file including pfdname")
    parser.add_argument('-t',type=int,help="The thread number")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = get_parse()
    t = time.time()
    if args.f is None:
        pred = predict()
    else:
        pred = predict(args.f)
    #pred.prof_svm_pred(model="trained_model/prof_svm.pkl")
    print("find %d pfd in %.2f"%(len(pred.pfds),time.time()-t))
    print("start.")
    if args.t is None:
        num_worker = os.cpu_count() - 1
    else:
        num_worker = args.t
    pred.topmodel_pred(CurDir+"/trained_model/top_lr0.1_newtrain1.model",   # model for Top LR layer
                        CurDir+"/trained_model/ann_512prof512dm_6000.pkl1",  # model for ANN module
                        CurDir+"/trained_model/3chs_coatnet29_newtrain.model",num_works=num_worker) # model for CoAtNet module
    #pred.topmodel_pred(CurDir+"/trained_model/top_lr_train4_t0.1_2.0.model",
    #                    CurDir+"/trained_model/ann_512prof512dm_train4_1.pkl",
    #                    CurDir+"/trained_model/ccct-train4-3chs-20.model",num_works=80)

    #pred.profdm_ann_pred("./trained_model/ann_profdm.pkl",scoretxt="new_test1_profdm.score",num_works=20,decimal=5)
