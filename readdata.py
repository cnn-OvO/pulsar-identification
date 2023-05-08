from prepfold import pfd
from cv2 import resize
from cv2 import INTER_LINEAR
import numpy as np
from pandas import read_csv
from os import sched_getaffinity
from multiprocessing import Pool
import argparse
from torch.utils.data import Dataset

class readpfd(pfd):
    def __init__(self):
        return None

    def get2Ddata(self,filename,resize_bin=128,centre=True):
        if not filename == "self":
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

    def get1Ddata(self,filename,resize_bin=64,centre=True):
        if not filename == "self":
            pfd.__init__(self,filename)

        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        sumprof = self.plot_sumprof()
        dmcurve = self.DM_curve()
        if centre:
            sumprof = self.shiftphase(sumprof)
        sumprof = np.squeeze(resize(sumprof,dsize=(resize_bin,1),interpolation=INTER_LINEAR))
        sumprof = self.normalize(sumprof)
        return sumprof, dmcurve

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
        sum_col = x.sum(axis = 0)
        _, num_col = x.shape
        max_index = np.argmax(sum_col)
        shift = num_col //2 - max_index
        x = np.roll(x, shift, axis = 1)
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
    def __init__(self):
        self.num_workers = len(sched_getaffinity((0))) -1
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
        return data




"""
class pfddataloader(object):
    def __init__(self, file, labeled=True):
        if labeled:
            if file.endswith('.txt'):
                with open(file) as f:
                    self.pfdnames = []
                    self.label = []
                    for line in f:
                        self.pfdnames.append(line.split()[0])
                        self.label.append(int(line.split()[1]))

            elif file.endswith('.csv'):
                self.pfdnames = read_csv(file, usecols=[1])
                self.label = read_csv(file, usecols=[2])

            else:
                raise NameError('You must input a txtfile(endwith \'.txt\') or csvfile(endwith\'.csv\').')
        else:
            if file.endswith('.txt'):
                with open(file) as f:
                    self.pfdnames = []
                    for line in f:
                        self.pfdnames.append(line.split()[0])
            elif file.endswith('.csv'):
                self.pfdnames = read_csv(file, usecols=[1])
            else:
                raise NameError('You must input a txtfile(endwith \'.txt\') or csvfile(endwith\'.csv\').')
            self.label = [1] * len(self.pfdnames)
        self.num = len(self.label)

        
    def extractdata(self):
        num_workers = len(sched_getaffinity((0))) - 1
        print("loading data ...")
        with Pool(processes=num_workers) as pool:
            self.data = pool.map(readpfd().getdata,self.pfdnames)
        pool.close()
        pool.join() 
        #print(self.data)
        #print(np.array([[i[3]] for i in self.data]))
        #print(np.squeeze(np.array([[i[1]] for i in self.data])))
        return np.squeeze(np.array([i[0] for i in self.data])), np.squeeze(np.array([i[1] for i in self.data]))

    def extractoridata(self):
        num_workers = len(sched_getaffinity((0))) - 1
        print("loading data ...")
        with Pool(processes=num_workers) as pool:
            self.data = pool.map(readpfd().getunsizeddata,self.pfdnames)
        pool.close()
        pool.join() 
        return np.squeeze(np.array([i[0] for i in self.data])), np.squeeze(np.array([i[1] for i in self.data]))
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
    
