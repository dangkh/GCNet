import os
import json
from typing import List
import torch
import numpy as np
import random
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
from data.base_dataset import BaseDataset
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint

def extractID(s):
    s = s.split('_')[-1]
    s = s.split('$')[-1]
    return int(s)

def refineID_DIAG(s):
    r = []
    for ii in s:
        r.append(np.searchsorted(np.sort(s), ii))
    return r

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return: Sn [alldata_len, view_num]
    """
    # print (f'==== generate random mask ====')
    one_rate = 1 - missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum=2] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num)) # [samplenum, viewnum=2] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    error = 1
    while error >= 0.007:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        # print (f'first ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        try:
            ratio = one_num_iter / (view_num * alldata_len)
            ratio = int(ratio * 100)
        except Exception as e:
            print (f'second ratio: {ratio}, force to equal 1')
            ratio = 100

        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < ratio).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # print (f'third ratio: {ratio}')
        error = abs(one_rate - ratio)
        
    return matrix

class CMUMOSIMultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        # record & load basic settings 
        cvNo = opt.cvNo
        seed_everything(opt.seed)
        self.mask_rate = opt.mask_rate
        # self.adim = 3700
        # self.tdim = 38400
        # self.vdim = 1750
        # self.adim = 250
        # self.tdim = 50
        # self.vdim = 1000
        ## gain video feats
        self.max_len = -1
        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}
        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}
        inputData = pickle.load(open('../dataset/CMUMOSI/CMU_MOSI_full.pkl', 'rb'), encoding='latin1')
        self.train, self.valid, self.test = inputData['train'], inputData['valid'], inputData['test']
        self.data = self.train
        if set_name == 'val':
            self.data = self.valid
        elif set_name == 'tst':
            self.data = self.test
        self.raw_text, self.audio, self.vision, self.id, self.text, self.text_bert, \
        self.annotations, self.class_labels, self.regress_label = [self.data[v] for ii, v in enumerate(self.data)]
        self.text_bert = self.text_bert[:,0,:]
        self.max = 200
        self.Vid = [x.split('$')[0] for x in self.id]
        self.idsentence = [extractID(x) for x in self.id]
        self.numDiaglouge = 0
        self.listNumNode = []
        self.listID_diaglouge = []
        self.uniqueVid = []
        current = 0
        while current < len(self.Vid):
            self.uniqueVid.append(self.Vid[current])
            numNode = 1
            id_diaglouge = [self.idsentence[current]]
            while ((current + numNode < len(self.Vid) and (self.Vid[current+numNode] == self.Vid[current]))):
                id_diaglouge.append(self.idsentence[current+numNode])
                numNode+=1
            self.numDiaglouge += 1
            self.listNumNode.append(numNode)
            id_diaglouge = refineID_DIAG(id_diaglouge)
            self.listID_diaglouge.append(id_diaglouge)
            current += numNode
            if self.numDiaglouge > self.max:
                break

        # self.class_labels[np.where(self.class_labels == 2)] = 1
        self.startNode = [0]
        self.max_len = max(self.listNumNode)
        for i in range(len(self.listNumNode)):
            self.startNode.append(self.startNode[-1]+self.listNumNode[i])
        
        self.maskmatrix = random_mask(3, self.numDiaglouge, self.mask_rate) # [samplenum, view_num]

    def __getitem__(self, index):
        maskseq = self.maskmatrix[index] # (3, )
        missing_index = torch.LongTensor(maskseq)

        l,r = self.startNode[index], self.startNode[index+1]
        audio = self.audio[l:r]
        vision = self.vision[l:r]
        text = self.text[l:r]
        labels = self.regress_label[l:r]
        id_diaglouge = np.asarray(self.listID_diaglouge[index])
        rearrangeID = [np.where(id_diaglouge == ii)[0][0] for ii in range(len(id_diaglouge))]
        rearrangeAudio = [audio[idx,:] for idx in rearrangeID]
        audio = np.stack(rearrangeAudio, axis=0)
        A_feat = audio.reshape(len(audio), -1)

        rearrangeVision = [vision[idx,:] for idx in rearrangeID]
        vision = np.stack(rearrangeVision, axis=0)
        V_feat = vision.reshape(len(vision), -1)

        rearrangeText = [text[idx,:] for idx in rearrangeID]
        text = np.stack(rearrangeText, axis=0)
        L_feat = text.reshape(len(text), -1)
        rearrangeLabel = [labels[idx] for idx in rearrangeID]
        labels = np.stack(rearrangeLabel, axis=0)

        return {
            'A_feat': torch.FloatTensor(A_feat), 
            'V_feat': torch.FloatTensor(V_feat),
            'L_feat': torch.FloatTensor(L_feat),
            'label': torch.FloatTensor(labels),
            'missing_index': missing_index
        }
    
    def __len__(self):
        print(self.numDiaglouge)
        return self.numDiaglouge
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    # def collate_fn(self, batch):
    #     A = [sample['A_feat'] for sample in batch]
    #     V = [sample['V_feat'] for sample in batch]
    #     L = [sample['L_feat'] for sample in batch]
    #     lengths = torch.tensor([len(sample) for sample in A]).long()
    #     A = pad_sequence(A, batch_first=True, padding_value=0)
    #     V = pad_sequence(V, batch_first=True, padding_value=0)
    #     L = pad_sequence(L, batch_first=True, padding_value=0)
    #     label = torch.tensor([sample['label'] for sample in batch])
    #     int2name = [sample['int2name'] for sample in batch]
    #     return {
    #         'A_feat': A, 
    #         'V_feat': V,
    #         'L_feat': L,
    #         'label': label,
    #         'lengths': lengths,
    #         'int2name': int2name
    #     }

if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = MSPMultimodalDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    print('Reading from dataloader:')
    x = [a[100], a[34], a[890]]
    print('each one:')
    for i, _x in enumerate(x):
        print(i, ':')
        for k, v in _x.items():
            if k not in ['int2name', 'label']:
                print(k, v.shape)
            else:
                print(k, v)
    print('packed output')
    x = a.collate_fn(x)
    for k, v in x.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    