from __future__ import division
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D
from keras.layers.merge import dot, multiply
from keras.callbacks import ModelCheckpoint
import os
import pdb 
import json
import pickle as pkl 
import numpy as np
from tqdm import tqdm
import random

class CNN():

    def __init__(self, user_hist, mat, read_hist, nwords, embed_size):

        self.user_hist = json.load(open(user_hist))
        self.users = self.user_hist.keys()
        self.mat = pkl.load(open(mat))
        self.articles = self.mat.keys()
        self.read_hist = read_hist
        self.nwords = nwords
        self.embed_size = embed_size
        self.model = None

    def create_model(self):

        user_input = Input(shape=(1, self.read_hist, self.nwords, self.embed_size))
        article_input = Input(shape=(1, self.nwords, self.embed_size))

        conv3d = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(user_input)
        maxpooling3d = MaxPooling3D(pool_size=(2, 2, 2))(conv3d)
        conv3d2 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(maxpooling3d)
        maxpooling3d2 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d2)

        flatten3d = Flatten()(maxpooling3d2)

        conv2d = Conv2D(64, kernel_size=(3, 3), activation='relu')(article_input)
        maxpooling2d = MaxPooling2D(pool_size=(2, 2))(conv2d)
        conv2d2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxpooling2d)
        maxpooling2d2 = MaxPooling2D(pool_size=(2, 2))(conv2d2)
        
        flatten2d = Flatten()(maxpooling2d2)

        element_wise_product = multiply([flatten3d, flatten2d])
        fc = Dense(128, activation='relu')(element_wise_product)
        output = Dense(1, activation='sigmoid')(fc)

        self.model = Model(inputs=[user_input, article_input], outputs = output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


    def fit_model(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1)

    def get_model_summary(self):
        print self.model.summary()


def train():

    read_hist = 10
    negs = 2
    model = CNN('../data/user_history.json', '../data/mat.pkl', read_hist, 20, 300)
    model.create_model()
    model.get_model_summary()
   
    user_in = []
    article_in = []
    truth = []
    for user in model.users:
        if len(model.user_hist[user]) < 10 or len(model.user_hist[user]) > 15:
            continue
        temp = []
        user_hist = model.user_hist[user][:-1]
        
        if len(user_hist) > read_hist:
            for art in user_hist[:read_hist]:
                temp.append(model.mat[str(art)])
            for art in user_hist[read_hist:]:
                article_in.append(model.mat[str(art)])
                user_in.append(temp)
                truth.append(1)
                for i in range(negs):
                    article_in.append(model.mat[random.choice(model.articles)])
                    user_in.append(temp)
                    truth.append(0)
 
        else:
            for art in user_hist[:-1]:
                temp.append(model.mat[str(art)])
            for i in range(read_hist-len(temp)):
                temp.append(np.zeros((20,300)))
            article_in.append(model.mat[str(user_hist[-1])])
            user_in.append(temp)
            truth.append(1)
            for i in range(negs):
                article_in.append(model.mat[random.choice(model.articles)])
                user_in.append(temp)
                truth.append(0)

    user_in = np.array(user_in)
    article_in = np.array(article_in)
    user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
    article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])

    model.fit_model([user_in, article_in], np.array(truth), '10_20')

def test():
   
    read_hist = 10
    model = CNN('../data/user_history.json', '../data/mat.pkl', read_hist, 20, 300)
    model.create_model()
    model.get_model_summary()
    model.model.load_weights('../weights/10_20/weights-11-0.19.hdf5')
    
    hr = [0]*10
    ndcg = [0]*10
    ct = 0

    for user in tqdm(model.users):
        if len(model.user_hist[user]) < 10 or len(model.user_hist[user]) > 15:
            continue
        ct += 1
        temp = []
        user_hist = model.user_hist[user][:-1]
        user_in = []
        article_in = []
        
        if len(user_hist) > read_hist:
            for art in user_hist[:read_hist]:
                temp.append(model.mat[str(art)])
        else:
            for art in user_hist[:-1]:
                temp.append(model.mat[str(art)])
            for i in range(read_hist-len(temp)):
                temp.append(np.zeros((20,300)))
            
        for i in range(99):
            user_in.append(temp)
            article_in.append(model.mat[random.choice(model.articles)])
        user_in.append(temp)
        article_in.append(model.mat[str(user_hist[-1])])

        user_in = np.array(user_in)
        article_in = np.array(article_in)
        user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
        article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])
    
        out = model.model.predict([user_in, article_in])

        sorted_items = sorted(range(len(out)),key=lambda x:out[x])
        sorted_items.reverse()

        for k in range(10):
            rec = sorted_items[:k+1]
            if 99 in rec:
                hr[k] += 1
            for pos in range(k+1):
                if rec[pos] == 99: 
                    ndcg[k] += 1 / np.log2(1+pos+1)
        print ct, hr[9]
        
    HR = []
    NDCG = []

    for k in range(10):
        print k, 'hr',  hr[k], 'ndcg', ndcg[k]
        HR.append(float(hr[k]) / float(len(model.users)))
        NDCG.append(float(ndcg[k]) / float(len(model.users)))
        print k, 'HR',  HR[k], 'NDCG', NDCG[k]

if __name__ == "__main__":
    train()
