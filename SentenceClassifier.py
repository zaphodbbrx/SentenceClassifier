import numpy as np
import re
import pymystem3
#from gensim.models.fasttext import FastText
import pickle
from keras.models import Sequential
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from nltk.tokenize import TreebankWordTokenizer


class SentenceClassifier():
    def __init__(self):
        self.__tokenizer = TreebankWordTokenizer()
        #self.stemmer = SnowballStemmer('russian')
        self.__mystem = pymystem3.Mystem()
        self.__ft_c = pickle.load(open('pickles/ft_compressed.pkl', 'rb'))
        self.__class_names = pickle.load(open('pickles/class_names.pkl','rb'))
        self.__old2new = pickle.load(open('pickles/old2new.pkl','rb'))
        self.__new2old = pickle.load(open('pickles/new2old.pkl','rb'))
        self.__num2title = pickle.load(open('pickles/num2title.pkl','rb'))
        self.__build_net(mtype = 'cnn')

    
    def __make_padded_sequences(self, docs, max_length,w2v):
        tokens = [doc.split(' ') for doc in docs]
        vecs = [[w2v[t] if t in w2v else np.zeros(25) for t in ts] for ts in tokens]
        seqs = np.array([np.pad(np.vstack(v),mode = 'constant', pad_width = ((0,max_length-len(v)),(0,0))) if len(v)<max_length else np.vstack(v)[:max_length,:] for v in vecs])
        return seqs

    
    def __clean_message(self,text):
        text = str(text)
        if len(text)>0:
            text = re.sub('\W|\d',' ',text).lower()
            tokens = self.__tokenizer.tokenize(text)
            tokens = [self.__mystem.lemmatize(t)[0] for t in tokens]
            return ' '.join(tokens)
        
    def __build_net(self, mtype):
        if mtype == 'cnn':
            model = Sequential()
            model.add(Conv1D(filters = 42, kernel_size = 2, input_shape = (50,25), activation='relu'))
            model.add(Conv1D(filters = 20, kernel_size = 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(4))
            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(19, activation='softmax'))
            self.__net = model
            self.__net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
            self.__net.load_weights('weights/cnn.hdf5')
        elif mtype == 'lstm':
            model = Sequential()
            model.add(Bidirectional(LSTM(25, activation='relu'), input_shape = (None, 25)))
            model.add(BatchNormalization())
            model.add(Dense(19, activation='softmax'))
            self.__net = model
            self.__net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
            self.__net.load_weights

    def __eval_net(self, message):
        message = self.__clean_message(message)
        vecs = self.__make_padded_sequences([message], 50,self.__ft_c)
        prediction = self.__net.predict(vecs)
        return prediction#np.argmax(prediction)

    def predict(self, message):
        prediction = self.__eval_net(message)
        proba = np.max(prediction)
        result = self.__class_names[self.__new2old[np.argmax(prediction)]-1]
        #print('Класс: \n{}\nВероятность: \n{:1.3f}'.format(result, proba))
        return {
                'decision': result,
                'confidence': proba
                }

if __name__ == '__main__':
    while True:
        sc = SentenceClassifier()
        tm = input()
        sc.predict(tm)