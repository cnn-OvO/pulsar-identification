from prepfold import pfd
from cv2 import resize
from cv2 import INTER_LINEAR
import numpy as np
from pandas import read_csv
from os import sched_getaffinity
from multiprocessing import Pool
import argparse
from torch.utils.data import Dataset
import os
import torch
from torchvision.transforms.functional import rotate
import math

class readpfd(pfd):
    def __init__(self):
        return None

    def getalldata(self,filename,resize_bin=64,centre=True):
        if not filename == 'self':
            pfd.__init__(self,filename)
        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        ddm = (self.dms.max() - self.dms.min())/2.
        loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
        loDM = max((0, loDM)) #make sure cut off at 0 DM
        hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
        dm_plot = self.plot_chi2_vs_DM(loDM,hiDM,N=4096)[0]
        dm_plot = self.normalize(dm_plot)
        dm_plot = np.reshape(dm_plot,(resize_bin,resize_bin))
        fvsp = self.plot_subbands()
        tvsp = self.plot_intervals()
        tvsp = self.plot_intervals()
        if centre:
            fvsp = self.shiftphase(fvsp)
            tvsp = self.shiftphase(tvsp)
        fvsp = resize(fvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        tvsp = resize(tvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        fvsp = self.normalize(fvsp)
        tvsp = self.normalize(tvsp)

        dm_curve = self.plot_chi2_vs_DM(loDM,hiDM,N=200)[0]
        dm_curve = self.normalize(dm_curve)
        sumprof = self.plot_sumprof()
        if centre:
            sumprof = self.shiftphase(sumprof)
        sumprof = np.squeeze(resize(sumprof,dsize=(1,resize_bin),interpolation=INTER_LINEAR))
        sumprof = self.normalize(sumprof)
        return np.array((fvsp,tvsp,dm_plot)), np.concatenate((sumprof, dm_curve),axis=0)


    def get3Chansdata(self,filename,resize_bin=64,centre=True):
        pfd.__init__(self,filename)
        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        ddm = (self.dms.max() - self.dms.min())/2.
        loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
        loDM = max((0, loDM)) #make sure cut off at 0 DM
        hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
        dm_plot = self.plot_chi2_vs_DM(loDM,hiDM,N=4096)[0]
        dm_plot = self.normalize(dm_plot)
        dm_plot = np.reshape(dm_plot,(resize_bin,resize_bin))
        fvsp = self.plot_subbands()
        tvsp = self.plot_intervals()
        if centre:
            fvsp = self.shiftphase(fvsp)
            tvsp = self.shiftphase(tvsp)
        fvsp = resize(fvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        tvsp = resize(tvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        fvsp = self.normalize(fvsp)
        tvsp = self.normalize(tvsp)

        return np.array((fvsp,tvsp,dm_plot))

    def get2Ddata(self,filename,resize_bin=64,centre=True):
        pfd.__init__(self,filename)
        
        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        fvsp = self.plot_subbands()
        tvsp = self.plot_intervals()
        if centre:
            fvsp = self.shiftphase(fvsp)
            tvsp = self.shiftphase(tvsp)
        fvsp = resize(fvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        tvsp = resize(tvsp, dsize=(resize_bin,resize_bin),interpolation=INTER_LINEAR)
        fvsp = self.normalize(fvsp)
        tvsp = self.normalize(tvsp)
        return np.array((fvsp,tvsp))

    def get1Ddata(self,filename,prof_bin=512,dm_bins=512,centre=True):
        if not filename == "self":
            pfd.__init__(self,filename)

        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        ddm = (self.dms.max() - self.dms.min())/2.
        loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
        loDM = max((0, loDM)) #make sure cut off at 0 DM
        hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
        dm = self.plot_chi2_vs_DM(loDM,hiDM,N=dm_bins)[0]
        dm = self.normalize(dm)
        sumprof = self.plot_sumprof()
        #dmcurve = self.plot_chi2_vs_DM(N=200)
        if centre:
            sumprof = self.shiftphase(sumprof)
        sumprof = np.squeeze(resize(sumprof,dsize=(1,prof_bin),interpolation=INTER_LINEAR))
        sumprof = self.normalize(sumprof)
        return np.concatenate((sumprof, dm),axis=0)

    def get_sumprof(self,filename,resize_bin=256,centre=True):
        if not filename == "self":
            pfd.__init__(self,filename)

        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        sumprof = self.plot_sumprof()
        if centre:
            sumprof = self.shiftphase(sumprof)
        sumprof = np.squeeze(resize(sumprof,dsize=(1,resize_bin),interpolation=INTER_LINEAR))
        sumprof = self.normalize(sumprof)
        return sumprof

    def get_dmcurve(self,filename,resize_bin=200):
        if not filename == "self":
            pfd.__init__(self,filename)

        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        ddm = (self.dms.max() - self.dms.min())/2.
        loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
        loDM = max((0, loDM)) #make sure cut off at 0 DM
        hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
        dm = self.plot_chi2_vs_DM(loDM,hiDM,N=resize_bin)[0]
        #dm = self.DM_curve()
        dm = self.normalize(dm)
        return dm

    def getunsizeddata(self, filename,):
        if not filename == "self":
            pfd.__init__(self, filename)
        self.dedisperse(doppler=1)
        self.adjust_period()
        fvsp = self.plot_subbands()
        fvsp = self.shiftphase(fvsp)
        fvsp = self.normalize(fvsp)
        sumprof = self.plot_sumprof()
        tvsp = self.plot_intervals()
        tvsp = self.shiftphase(tvsp)
        tvsp = self.normalize(tvsp)
        dmcurve = self.DM_curve()
        return sumprof, dmcurve, fvsp, tvsp

    def normalize(self,x):
        x = ( x - x.min()) / (x.max() - x.min())
        return x
    
    def shiftphase(self,x):
        if len(x.shape) == 2:
            sum_col = x.sum(axis = 0)
            _, num_col = x.shape
            max_index = np.argmax(sum_col)
            shift = num_col //2 - max_index
            x = np.roll(x, shift, axis = 1)

        elif len(x.shape) == 1:
            sum_col = x
            num_col = x.shape[0]
            max_index = np.argmax(sum_col)
            shift = num_col //2 - max_index
            x = np.roll(x, shift, axis = 0)
        else :
            raise ValueError("shiftphase only process 1-Dim or 2-Dim array")
        return x

    def standarize(self,fvsp,tvsp,normalize=True,resize_bins=224):

        fvsp = resize(fvsp, dsize=(resize_bins,resize_bins), interpolation=INTER_LINEAR)
        tvsp = resize(tvsp, dsize=(resize_bins,resize_bins), interpolation=INTER_LINEAR)
        fvsp = self.shiftphase(fvsp)
        tvsp = self.shiftphase(tvsp)
        sumprof = fvsp.sum(axis=1)
        if normalize:
            fvsp = self.normalize(fvsp)
            tvsp = self.normalize(tvsp)
            sumprof = self.normalize(sumprof)
        
        return sumprof, fvsp, tvsp

class pfddataloader(object):
    def __init__(self, num_workers = -1):
        if num_workers == -1 :
            self.num_workers = len(sched_getaffinity((0))) -1
        else:
            self.num_workers = num_workers
        return None

    def loader2Ddata(self, pfdlist, label=None):
        with Pool(processes=self.num_workers,) as pool:
            data = pool.map(readpfd().get2Ddata, pfdlist)
        pool.close()
        pool.join()
        return np.array(data)
    
    def loader1Ddata(self, pfdlist, label=None):
        with Pool(processes=self.num_workers,) as pool:
            data = pool.map(readpfd().get1Ddata, pfdlist)
        pool.close()
        pool.join()
        return np.array(data)

    def loadersumprof(self, pfdlist, label=None):
        with Pool(processes=self.num_workers,) as pool:
            data = pool.map(readpfd().get_sumprof, pfdlist)
        pool.close()
        pool.join()
        return np.array(data)

    def loaderdmcurve(self, pfdlist, label=None):
        with Pool(processes=self.num_workers,) as pool:
            data = pool.map(readpfd().get_dmcurve, pfdlist)
        pool.close()
        pool.join()
        return np.array(data)


class MYDataset(Dataset):
    def __init__(self, txtfile, dir, channel = 2):
        self.dir = dir
        self.pfdname = []
        self.label = []
        self.channel = channel
        with open(txtfile) as f:
            for line in f:
                line = line.split()
                self.pfdname.append(line[0])
                self.label.append(line[1])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        pfdpath = os.path.join(self.dir, self.pfdname[idx])
        if self.channel == 2:
            data = torch.as_tensor(readpfd().get2Ddata(pfdpath,resize_bin=64),dtype=torch.float32)
        elif self.channel == 3:
            data = torch.as_tensor(readpfd().get3Chansdata(pfdpath,resize_bin=64),dtype=torch.float32)
        elif self.channel == 4:
            data = torch.as_tensor(readpfd().get2Ddata(pfdpath,resize_bin=64),dtype=torch.float32)
            data2 = rotate(data,angle=90)
            data = torch.cat([data,data2],dim=0)
            del data2
        else: 
            pass
        label = torch.as_tensor(int(self.label[idx]),dtype=torch.int64)
        return data, label


def readtxt(txtfile, num_works, data_type = "1Ddata"):
    with open(txtfile, 'r') as f:
        pfd = []
        label= []
        for line in f:
            line = line.split()
            pfd.append(line[0])
            label.append(int(line[1]))
    rp = readpfd()
    if data_type == "1Ddata":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get1Ddata, pfd)
    elif data_type == "profile":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get_sumprof, pfd)
    elif data_type == "dm":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get_dmcurve, pfd)
    X = np.array(X)
    label = np.array(label)
    return pfd, X, label

def load_pred_data(pfd, num_works, data_type = "1Ddata"):
    #with open(txtfile, 'r') as f:
    #    pfd = []
    #    for line in f:
    #        line = line.split()
    #        pfd.append(line[0])
    if len(pfd) < num_works:
        num_works = 1
    rp = readpfd()
    if data_type == "1Ddata":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get1Ddata, pfd)
    elif data_type == "profile":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get_sumprof, pfd)
    elif data_type == "dm":
        with Pool(processes = num_works) as pool:
            X = pool.map(rp.get_dmcurve, pfd)
    X = np.array(X)
    return X

def from_idx_read(pfds, num_works = 1, channel = 3):
    if len(pfds) < 5*num_works:
        num_works = math.ceil(len(pfds) / 5)
    rp = readpfd()
    if channel == 2:
        with Pool(processes = num_works) as pool:
            data = np.array(pool.map(rp.get2Ddata,pfds))
        pool.close()
        pool.join()
    elif channel == 3:
        with Pool(processes = num_works) as pool:
            data = np.array(pool.map(rp.get3Chansdata, pfds))
        pool.close()
        pool.join()
    else: 
        pass
    data = torch.tensor(data,dtype=torch.float32)
    return data

def outputanalysis(outputs, labels, threshold = 0.1):
    count1 = 0
    count2 = 0
    #print(len(outputs))
    #print(len(labels))
    if len(outputs.shape) != 1:
        for i in range(len(labels)):
            if labels[i]==1 and outputs[i,1]>threshold:
                count1 += 1
            elif labels[i]==0 and outputs[i,1]>=threshold:
                count2 += 1
        #outputs = np.argmax(outputs,axis=1)
    elif len(outputs.shape) == 1:
        for i in range(len(labels)):
            if labels[i]==1 and outputs[i]>threshold:
                count1 += 1
            elif labels[i]==0 and outputs[i]>=threshold:
                count2 += 1
        #outputs = np.argmax(outputs)
    #print("real pulsar/all:%d/%d"%(labels.sum(),len(labels)))
    #outputs = np.argmax(outputs,axis=1)
    recall = count1/labels.sum()
    false_positive_rate = count2/(len(labels)-labels.sum())
    #print("recall:%d/%d=%f"%(count1,labels.sum(),recall))
    #print("false positive rate:%d/%d=%f"%(count2,(len(labels)-labels.sum()),count2/(len(labels)-labels.sum())))
    #labels = np.argmax(labels,axis=1)
    count = labels.sum() - count1 + count2
    accuracy =  1 - ( count / len(labels) )
    #print("acc=%f"%(accuracy))
    f1 = 2*accuracy*recall/(accuracy+recall)
    #np.save(np.concatenate('output'(pred,outputs,log_prob,labels)), )
    return recall, false_positive_rate, accuracy, f1
"""
class AllDataset(Dataset):
    def __init__(self, txtfile, dir):
        self.dir = dir
        self.pfdname = []
        self.label = []
        with open(txtfile) as f:
            for line in f:
                line = line.split()
                self.pfdname.append(line[0])
                self.label.append(line[1])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        pfdpath = os.path.join(self.dir, self.pfdname[idx])
        data = 
        label = self.label[idx]
        label = torch.as_tensor(int(self.label[idx]),dtype=torch.int64)
        return data, label
"""


def get_parse():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('file',type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parse()
    ldr = pfddataloader(args.file)
    A = ldr.extractdata()
    print(A[1].shape)
    
