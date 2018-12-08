import pickle as pkl
import re

#funções auxiliares
def extract_info(word):
    word = word.split()
    ans1 = word[1].lstrip("[").rstrip("]")
    word = list(filter(lambda x: re.match("^[a-zA-ZàÀéÉóÓ]", x) != None, word))
    if (len(word) < 2):
        ans2 = "UNK"
    else:
        ans2 = word[1]
    return [ans1, ans2]

def filter_sentence(sent):
    return (re.match("^[a-zA-ZàÀéÉóÓ]", sent) != None)

#pega dataset cru
dataset = pkl.load(open("sentlist.pkl", "rb"))
print("loaded")

#extracao de info: limpa sentencas
dataset = list(map(lambda x: list(filter(filter_sentence, x)), dataset))

#extracao de info: transforma cada palavra em [palavra, tag]
dataset = list(map(lambda sent: list(map(extract_info, sent)), dataset))

print("info extracted")

#tratemnto de dataset: elimina sentencas pequenas
dataset = list(filter(lambda x: len(x) > 2, dataset))
print("filtered")

#tratamento de dataset: números viram NUM
for i, sent in enumerate(dataset):
    for j, word in enumerate(sent):
        try:
            float(word[0].replace(".", "").replace(",", "").replace("/", "").replace("=", "").replace("-", "").replace(":", "").replace("ª", "").replace("º", ""))
        except ValueError:
            continue
        else:
            dataset[i][j][0] = "NUM"

print("numbered")

#tratamento de dataset: horarios viram HOR
for i, sent in enumerate(dataset):
    for j, word in enumerate(sent):
        if re.match("[0-9]+h[0-9]+", word[0]):
            dataset[i][j][0] = "TIME"

print("timed")

#tratamento de etiquetas: eliminar etiquetas raras
labelfreq = dict()
for sent in dataset:
    for word in sent:
        if word[1] in labelfreq:
            labelfreq[word[1]] += 1
        else:
            labelfreq[word[1]] = 1

for i, sent in enumerate(dataset):
    for j, word in enumerate(sent):
        if labelfreq[word[1]] < 10000:
            dataset[i][j][1] = "UNK"


print("labelfreqed")

#calcula ocorrencias de palavra
wordoc = dict()
for sent in dataset:
    for word in sent:
        oc = word[0]
        if oc in wordoc:
            wordoc[oc] += 1
        else:
            wordoc[oc] = 1


for i, sent in enumerate(dataset):
    for j, word in enumerate(sent):
        if wordoc[word[0]] <= 5:
            dataset[i][j][0] = "UNK"

print("wordoced")

pkl.dump(dataset, open("sentlist_sanitized.pkl", "wb"))

print("saved")
