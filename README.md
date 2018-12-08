# Artigo 2 - Métodos Neurais Para Etiquetadores Morfossintáticos
Antonio Augusto Abello - NUSP: 8536152

Aplicação de Word2Vec, LSTM e Bidirectional LSTM em um dataset público de etiquetagem
morfossintática em portugûes (CETENFolha) disponibilizado pela Linguateca (https://www.linguateca.pt/CETENFolha/) 
utilizando Keras-Tensorflow

## Entrega:
Esse repositório contém

* README.md - este arquivo, descrevendo a entrega e o uso dos programas
* artigo2.pdf - O artigo descrevendo o programa, experimentos e resultados
* \*.py - scripts em pythons descritos no artigo 

## Dependências:
Python3 e bibliotecas padrão (Pickle)
Numpy (1.15.2), Tensorflow (1.6.0), Keras (2.2.4), Scikit-Learn (0.20.0)   

## Modo de Uso:
python3 pre\_proc.py <arquivo_de_entrada>
Recebe como entrada o arquivo cru tal como disponibilizado e produz "sentlist.pkl" contendo
uma lista de sentenças (cada sentença sendo em si uma lista de strings correspondendo às linhas)

## python3 sanitize.py
Sanitiza e prepara dataset conforme descrito no artigo. Produz "sentlist\_sanitized.pkl"

## python3 generate\_unsupervised\_dataset.py
Pega o dataset sanitizado e gera um dataset no formato correto para aprendizado de embedding
word2vec. Gera "conversor.pkl", que é um dicionário duplo palavra <-> inteiro identificador.
O inteiro identificador é baseado na ordem de maior número de ocorrências. Gera 
"unsupervised\_dataset.npz"

## python3 generate\_supervised\_dataset.py
Pega o dataset sanitizado e gera um dataset no formato correto para aprendizado de um classificador.
Palavras são convertidas em inteiros utilizando "conversor.pkl" e etiquetas são convertidas
em um vetor one-hot do tamanho do número de etiquetas disponíveis. Gera "supervised\_dataset.npz"

## python3 tran\_word2vec.py <ndim>
Utiliza "conversor.pkl" e "unsupervised\_dataset.npz
Treina um modelo de word2vec, com dimensão "ndim". Gera "loss\_hist<ndim>" com informações sobre
a evolução da função de perda com o tempo. Também gera "word2vec_<ndim>", modelo do Keras que
pode ser utilizado nas outras partes do programa

## python3 train\_postagger.py
Treina um modelo de LSTM (default: bidirecional) para etiquetagem morfossintática utilizando
"supervised\_dataset.npz". Testa em cada epoch (default: 2) no dataset de teste


Para mais informações sobre os programas checar o arquivo PDF
