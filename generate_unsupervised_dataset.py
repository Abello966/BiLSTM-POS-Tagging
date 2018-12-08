import numpy as np
import tensorflow as tf
import keras as kr
import pickle as pkl
import operator

import keras.preprocessing.sequence as sequence

data = pkl.load(open("sentlist_sanitized.pkl", "rb"))
print("Loaded")
data = list(map(lambda sent: list(map(lambda word: word[0], sent)), data))
print("Trimmed")


#conta ocorrencias para definir ordem
wordoc = dict()
for sent in data:
    for word in sent:
        if word in wordoc:
            wordoc[word] += 1
        else:
            wordoc[word] = 1

vocab_size = len(wordoc.keys()) + 1
print("Wordoced")

#gera ordem e identificação unica palavra <> numero
wordsorted = sorted(wordoc.items(), key = operator.itemgetter(1), reverse=True)
conversor = dict()

for i, word in enumerate(wordsorted): 
    conversor[i + 1] = word[0]
    conversor[word[0]] = i + 1 #"by convention index 0 is a non-word" 

#converte dataset em representação one-hot/inteiro
for i, sent in enumerate(data):
    for j, word in enumerate(sent):
        data[i][j] = conversor[data[i][j]]

print("converted")

#vamos precisar desse dicionário para proximas tarefas
pkl.dump(conversor, open("conversor.pkl", "wb"))

print("dumped conversor")

#economiza memória?
del conversor
del wordsorted
del wordoc


#gera dataset de skip-grams
table = sequence.make_sampling_table(vocab_size)
couples = []
labels = []

for sent in data:
    thiscoup, thislabel = sequence.skipgrams(sent, vocab_size, sampling_table=table, window_size=3)
    couples += thiscoup
    labels += thislabel

print("sampled skipgrams")

#termina de montar dataset
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")
labels = np.array(labels, dtype="int32")

#termina dataset
np.savez("unsupervised_dataset", word_target, word_context, labels)
print("saved dataset")


