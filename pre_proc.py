import pickle
import re 
import sys


if len(sys.argv) != 3:
    print("Uso: python3 pre_proc.py <arquivo_de_entrada>")
    sys.exit()
    

sentlist = []
myf = open(sys.argv[1])

while True:
    try:
        line = next(myf)
    except StopIteration:
        #Acabou o arquivo e não estamos no meio de uma sentença, adeus
        break
    
    if re.match("<s>", line):
        #start of a sentence
        newsent = []
        while True:
            try:
                line = next(myf)
            except StopIteration:
                #Acabou o arquivo no meio de uma sentença, descarta malformado
                del newsent
                break
                
            if re.match("</s>", line):
                #fim de sentenca
                sentlist.append(newsent)
                del newsent
                break
                
            elif re.match("<s>", line):
                #mal-formado, descarta tudo
                del newsent
                newsent = []
                
            else:
                newsent.append(line)

#economiza memoria?
myf.close()

outf = open("sentlist.pkl", 'wb')
pickle.dump(sentlist, outf)
outf.close()
