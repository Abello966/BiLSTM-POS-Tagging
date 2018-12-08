import sys
import pickle as pkl
import tensorflow as tf
import keras as kr
from keras.models import Model
from keras.callbacks import Callback
import numpy as np
import operator

import models

#define um callback custom para checar treinamento
class SimilarityCallback(Callback):
    def __init__(self, mapper, word, numex, numsim, model, logfile):
        self.mapper = mapper #dicionario numero <> palavra
        self.wordvalid = word #palavra a ser checada
        self.numex = numex #numero de numex palavras mais frequentes testadas
        self.numsim = numsim #numero de palavras mostradas
        self.model = model #rewrite model pelo validador
        self.logfile = logfile

    def on_batch_end(self, batch, logs=None):
        if (batch % 10 == 0):
            self.logfile.write(str(logs['loss']) + "\n")

    def on_epoch_begin(self, batch, logs=None):
        self.on_epoch_end(batch, logs=logs)

    def on_epoch_end(self, batch, logs=None):
        wordnum = self.mapper[self.wordvalid]
        wordsim = dict()
        for i in range(1, self.numex):
            if i != wordnum:
                sem = self.model.predict([np.array([wordnum]), np.array([i])])
                wordsim[i] = sem[0][0]

        topsim = sorted(wordsim.items(), key=operator.itemgetter(1))

        
        print("Palavras mais próximas de %s" % self.wordvalid, end=": ")
        for j in range(self.numsim):
            print(self.mapper[topsim[j][0]], end=", ")
        print()


if (len(sys.argv) != 2):
    print("Uso: %s <dimensao do word2vec>" % sys.argv[0])
    sys.exit(-1)
        
#dicionario duplo
conversor = pkl.load(open("conversor.pkl", "rb"))

try:
    dimension = int(sys.argv[1])
except ValueError:
    print("Uso: %s <dimensao do word2vec>" % sys.argv[0])
    sys.exit(-1)

#dataset
dataset = np.load("unsupervised_dataset.npz")
word_target = dataset['arr_0']
word_context = dataset['arr_1']
labels = dataset['arr_2']

valid_word = "PT"
vocab_size = max(np.max(word_target), np.max(word_context))
logfile = open("loss_hist" + str(dimension), "w")

#model com dois outputs: previsão de proximidade, semelhança
model, validator = models.word2vec_model(vocab_size, dimension)
model.compile(loss=["binary_crossentropy"], optimizer="adam")

#callback para falar semelhança com o tempo
sim_cb = SimilarityCallback(conversor, valid_word, 10000, 8, validator, logfile)
model.fit([word_target, word_context], [labels], epochs=2, batch_size=256, callbacks=[sim_cb])
model.save("word2vec" + str(dimension))
logfile.close()
