import numpy as np
import keras as kr
import pickle as pkl

from keras.models import load_model
from keras.utils.np_utils import to_categorical

#dados
data = pkl.load(open("sentlist_sanitized.pkl", "rb"))

#conversor palavra <> onehot
conversor = pkl.load(open("conversor.pkl", "rb"))

#precisamos de um conversor para labels também
labels = set()
for sent in data:
    for word in sent:
        labels.add(word[1])

labels = sorted(list(labels)) #numero de um label é sua ordem alfabetica

Xdata = []
ydata = [] 

for i, sent in enumerate(data):
    tens = np.linspace(0, len(data), 10).astype("int")
    ones = np.linspace(0, len(data), 100).astype("int")

    xin = list(map(lambda word: conversor[word[0]], sent))
    yin = list(map(lambda word: labels.index(word[1]), sent))        
    yin = to_categorical(yin, num_classes=len(labels))
    Xdata.append(xin)
    ydata.append(yin)

Xdata = np.array(Xdata)
ydata = np.array(ydata)
np.savez("supervised_dataset", Xdata, ydata)

