import os
import time
import glob
import tqdm
import pickle
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


## gain name2features [only one speaker]
## videoLabels: from [-3, 3], type=float
def read_data(label_path, feature_root):

    ## gain (names, speakers)
    names = []
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids = pickle.load(open(label_path, "rb"), encoding='latin1')
    for ii, vid in enumerate(videoIDs):
        uids_video = videoIDs[vid]
        names.extend(uids_video)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        feature = []
        feature_path = os.path.join(feature_root, name+'.npy')
        feature_dir = os.path.join(feature_root, name)
        if os.path.exists(feature_path):
            single_feature = np.load(feature_path)
            single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
            feature.append(single_feature)
            feature_dim = max(feature_dim, single_feature.shape[-1])
        else: ## exists dir, faces
            facenames = os.listdir(feature_dir)
            for facename in sorted(facenames):
                facefeat = np.load(os.path.join(feature_dir, facename))
                feature_dim = max(feature_dim, facefeat.shape[-1])
                feature.append(facefeat)
        # sequeeze features
        single_feature = np.array(feature).squeeze()
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim, ))
        elif len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)
        features.append(single_feature)

    ## save (names, features)
    print (f'Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; No. sample is {len(names)}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats = {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]

    return name2feats, feature_dim

def extractID(s):
    s = s.split('_')[-1]
    s = s.split('$')[-1]
    return int(s)

def refineID_DIAG(s):
    r = []
    for ii in s:
        r.append(np.searchsorted(np.sort(s), ii))
    return r


class CMUMOSIDataset(Dataset):

    def __init__(self, label_path, type):

        ## read utterance feats
        # name2audio, adim = read_data(label_path, audio_root)
        # name2text, tdim = read_data(label_path, text_root)
        # name2video, vdim = read_data(label_path, video_root)
        self.adim = 3700
        self.tdim = 38400
        self.vdim = 1750
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
        inputData = pickle.load(open('./features/aligned_50.pkl', 'rb'), encoding='latin1')
        self.train, self.valid, self.test = inputData['train'], inputData['valid'], inputData['test']
        self.data = self.train
        if type == 'valid':
            self.data = self.valid
        elif type == 'test':
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
        # for ii, vid in enumerate(sorted(self.videoIDs)):
        #     uids = self.videoIDs[vid]
        #     labels = self.videoLabels[vid]
        #     speakers = self.videoSpeakers[vid]
        #     self.max_len = max(self.max_len, len(uids))
        #     speakermap = {'': 0}
        #     self.videoAudioHost[vid] = []
        #     self.videoTextHost[vid] = []
        #     self.videoVisualHost[vid] = []
        #     self.videoAudioGuest[vid] = []
        #     self.videoTextGuest[vid] = []
        #     self.videoVisualGuest[vid] = []
        #     self.videoLabelsNew[vid] = []
        #     self.videoSpeakersNew[vid] = []
        #     for ii, uid in enumerate(uids):
        #         self.videoAudioHost[vid].append(name2audio[uid])
        #         self.videoTextHost[vid].append(name2text[uid])
        #         self.videoVisualHost[vid].append(name2video[uid])
        #         self.videoAudioGuest[vid].append(np.zeros((self.adim, )))
        #         self.videoTextGuest[vid].append(np.zeros((self.tdim, )))
        #         self.videoVisualGuest[vid].append(np.zeros((self.vdim, )))
        #         self.videoLabelsNew[vid].append(labels[ii])
        #         self.videoSpeakersNew[vid].append(speakermap[speakers[ii]])
        #     self.videoAudioHost[vid] = np.array(self.videoAudioHost[vid])
        #     self.videoTextHost[vid] = np.array(self.videoTextHost[vid])
        #     self.videoVisualHost[vid] = np.array(self.videoVisualHost[vid])
        #     self.videoAudioGuest[vid] = np.array(self.videoAudioGuest[vid])
        #     self.videoTextGuest[vid] = np.array(self.videoTextGuest[vid])
        #     self.videoVisualGuest[vid] = np.array(self.videoVisualGuest[vid])
        #     self.videoLabelsNew[vid] = np.array(self.videoLabelsNew[vid])
        #     self.videoSpeakersNew[vid] = np.array(self.videoSpeakersNew[vid])


    ## return host(A, T, V) and guest(A, T, V)
    def __getitem__(self, index):
        # vid = self.vids[index]
        l,r = self.startNode[index], self.startNode[index+1]
        audio = self.audio[l:r]
        vision = self.vision[l:r]
        text = self.text[l:r]
        labels = self.regress_label[l:r]
        id_diaglouge = np.asarray(self.listID_diaglouge[index])
        rearrangeID = [np.where(id_diaglouge == ii)[0][0] for ii in range(len(id_diaglouge))]
        rearrangeAudio = [audio[idx,:] for idx in rearrangeID]
        audio = np.stack(rearrangeAudio, axis=0)
        audio = audio.reshape(len(audio), -1)

        rearrangeVision = [vision[idx,:] for idx in rearrangeID]
        vision = np.stack(rearrangeVision, axis=0)
        vision = vision.reshape(len(vision), -1)

        rearrangeText = [text[idx,:] for idx in rearrangeID]
        text = np.stack(rearrangeText, axis=0)
        text = text.reshape(len(text), -1)
        rearrangeLabel = [labels[idx] for idx in rearrangeID]
        labels = np.stack(rearrangeLabel, axis=0)

        return torch.FloatTensor(audio),\
               torch.FloatTensor(text),\
               torch.FloatTensor(vision),\
               torch.FloatTensor(np.zeros((len(audio), self.adim, ))),\
               torch.FloatTensor(np.zeros((len(text),self.tdim, ))),\
               torch.FloatTensor(np.zeros((len(vision), self.vdim, ))),\
               torch.FloatTensor([0] * self.listNumNode[index]),\
               torch.FloatTensor([1]*len(labels)),\
               torch.FloatTensor(labels),\
               self.uniqueVid[index]


    def __len__(self):
        return self.numDiaglouge

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print (f'max seqlen: {self.max_len}')
        return self.max_len

    # def collate_fn(self, data):
    #     datnew = []
    #     dat = pd.DataFrame(data)
    #     for i in dat: # row index
    #         if i<=5: 
    #             datnew.append(pad_sequence(dat[i])) # pad
    #         elif i<=8:
    #             datnew.append(pad_sequence(dat[i], True)) # reverse
    #         else:
    #             datnew.append(dat[i].tolist()) # origin
    #     return datnew