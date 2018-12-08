import keras as kr
from keras.layers import Input, Reshape, Embedding, Dot, Dense, LSTM, Dropout, TimeDistributed
from keras.layers import Bidirectional
from keras.models import Model

def identity_loss(y_pred, y_true):
    return kr.backend.constant(0)

def word2vec_model(vocab_size, vector_dim):

    input_target = Input((1,))
    input_context = Input((1,))

    embedder = Embedding(vocab_size, vector_dim, input_length=1)

    target = embedder(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedder(input_context)
    context = Reshape((vector_dim, 1))(context)

    #usado para validacao
    similarity = Dot(0, normalize=True, name="similarity")([target, context])

    dot = Dot(1, normalize=False, name="outputer")([target, context])

    dot = Reshape((1,))(dot)

    output = Dense(1, activation='sigmoid')(dot)

    model = Model(inputs=[input_target, input_context], outputs=[output])

    validator = Model(inputs=[input_target, input_context], outputs=[similarity])

    return model, validator

def embedder(word2vec_mod):
    
    incoming = word2vec_mod.input[0]
    outgoing = word2vec_mod.get_layer("embedding_1").get_output_at(0)

    return Model(inputs=incoming, outputs=outgoing)


def LSTM_model(word2vec, emb_dim, class_size):
    ember = word2vec.get_layer("embedding_1")
    ember.trainable = False

    begin = Input((None, 1))
    embedded = TimeDistributed(ember)(begin)
    emberres = TimeDistributed(Reshape((emb_dim,)))(embedded)
    proc = LSTM(emb_dim, return_sequences=True)(emberres)
    drop = Dropout(0.5)(proc)
    out = TimeDistributed(Dense(class_size, activation="softmax"))(drop)

    model = Model(begin, out)
    return model

def biLSTM_model(word2vec, emb_dim, class_size):
    ember = word2vec.get_layer("embedding_1")
    ember.trainable = False

    begin = Input((None, 1))
    embedded = TimeDistributed(ember)(begin)
    emberres = TimeDistributed(Reshape((emb_dim,)))(embedded)

    proc = Bidirectional(LSTM(emb_dim, return_sequences=True), merge_mode='concat')(emberres)

    drop = Dropout(0.5)(proc)

    out = TimeDistributed(Dense(class_size, activation="softmax"))(drop)

    model = Model(begin, out)
    return model
     

    
    
     
