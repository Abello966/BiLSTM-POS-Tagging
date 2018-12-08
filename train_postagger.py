import keras as kr
import numpy as np
import sys
from sklearn.model_selection import train_test_split

import models

#gerador de exemplos para treino
class KerasBatchGenerator(object):

    def __init__(self, datax, datay, maxbatch=32):
        self.datax = datax
        self.datay = datay
        self.maxbatch = maxbatch

        lendict = dict()
        for i, sent in enumerate(datay):
            if sent.shape[0] in lendict:
                lendict[sent.shape[0]].append(i)
            else:
                lendict[sent.shape[0]] = [i]

        self.lendict = lendict

    def num_steps(self):
        ans = 0
        for i in self.lendict:
            ans += len(self.lendict[i]) // 32
            if len(self.lendict[i]) % 32 != 0:
                ans += 1

        return ans

    def generate(self):
        while True:
            for size in self.lendict:
                start = 0
                idxarray = self.lendict[size]
                while start < len(idxarray):
                    batchidx = idxarray[start:start + self.maxbatch]
                    batch_x = np.array(list(self.datax[batchidx]))
                    batch_x = batch_x.reshape(batch_x.shape + (1,))
                    batch_y = np.array(list(self.datay[batchidx]))
                    yield batch_x, batch_y
                    start += self.maxbatch


#dados importantes
if (len(sys.argv) != 4):
    print("Uso: python3 %s <ndim> <seed> <bidirectional>" % sys.argv[0])
    sys.exit(-1)

try:
    dimension = int(sys.argv[1])
except ValueError:
    print("Uso: python3 %s <ndim> <seed> <bidirectional>" % sys.argv[0])
    sys.exit(-1)

try:
    seed = int(sys.argv[2])
except ValueError:
    print("Uso: python3 %s <ndim> <seed> <bidirectional>" % sys.argv[0])
    sys.exit(-1)

try:
    bidirectional = int(sys.argv[3])
except ValueError:
    print("Uso: python3 %s <ndim> <seed> <bidirectional>" % sys.argv[0])
    sys.exit(-1)
    


#dataset
data = np.load("supervised_dataset.npz")
X = data['arr_0']
y = data['arr_1']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=seed)

train_data_generator = KerasBatchGenerator(Xtrain, ytrain, maxbatch=32)
test_data_generator = KerasBatchGenerator(Xtest, ytest, maxbatch=32)

w2v = kr.models.load_model("word2vec_" + str(dimension))
if (bidirectional == 0):
    postagger = models.LSTM_model(w2v, dimension, 13)
    print("Training a LSTM with w2v dimension %d and seed %d" % (dimension, seed))
else:
    postagger = models.biLSTM_model(w2v, dimension, 13)
    print("Training a biLSTM with w2v dimension %d and seed %d" % (dimension, seed))

postagger.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_accuracy'])
postagger.fit_generator(train_data_generator.generate(), train_data_generator.num_steps(), 2,
                        validation_data=test_data_generator.generate(),
                        validation_steps=test_data_generator.num_steps(), verbose=2)

