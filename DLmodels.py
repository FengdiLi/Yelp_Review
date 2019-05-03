import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing import text
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Bidirectional, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score, f1_score


def CleanText(string):
    '''
    String cleaning.
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r" \'s", "\'s", string)
    string = re.sub(r" \'ve", "\'ve", string)
    string = re.sub(r" n\'t", "n\'t", string)
    string = re.sub(r" \'re", "\'re", string)
    string = re.sub(r" \'d", "\'d", string)
    string = re.sub(r" \'ll", "\'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r" \(", "", string)
    string = re.sub(r" \)", "", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\n", " ", string)
    return string


class LoadData:

    def __init__(self, data_file, embed_path):
        self.data = pd.read_csv(data_file, sep='\t', index_col=0)
        # assign review samples to two classes using [0,4) and [4, 5] criteria
        self.data['class'] = (self.data['stars'] >= 4).astype(int)
        self.data = self.data[['text', 'class']]
        self.data['text'] = self.data['text'].apply(CleanText)
        self.embed_path = embed_path

    def WordEmbedding(self, max_len, max_features=3000, w2v_size=300):

        # tokenization
        tk = text.Tokenizer(num_words=max_features, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n',
                            split=" ")
        tk.fit_on_texts(self.data['text'])
        word_index = tk.word_index
        self.X = pad_sequences(tk.texts_to_sequences(self.data['text']), 200,
                               padding='post', truncating='post')
        self.y = self.data['class']
        del self.data
        # load google news pre-trained model
        w2v_model = KeyedVectors.load_word2vec_format(self.embed_path, binary=True)
        # vectorizization & padding
        # dictionary vector matrix
        w2v_matrix = np.zeros((len(word_index) + 1, w2v_size))
        for word, i in word_index.items():
            if word in w2v_model.vocab:
                w2v_matrix[i] = w2v_model[word]
        # embedding
        w2v_emb = Embedding(len(word_index) + 1, w2v_size, weights=[w2v_matrix],
                            input_length=max_len, trainable=False)
        return tk, w2v_emb

    def Load(self, max_len):

        self.tk, self.w2v_emb = self.WordEmbedding(max_len)
        np.random.seed(1)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, train_size=0.7)
        del self.X, self.y


def Eval(X, y, name, model, verbose=True,
         output_path='..model/'):
    '''
    Model evaluation
    :param X: Test data input
    :param y: Test data label
    :param model: Model
    :param tk: Tokenizer
    :param max_len: Padding maximum length
    :return: Accuracy and F1 scores
    '''
    print(f'{name} Model')
    X_test = X
    y_test = y
    y_pred = model.predict_classes(X_test)
    print(f'Test Accuracy:{accuracy_score(y_test, y_pred)}')
    print(f'Test F1:{f1_score(y_test, y_pred)}')
    if verbose:
        model.save(f'{output_path}{name}.h5')


def base_LSTM(X, y, w2v_emb, dropout=0.2,
              loss='binary_crossentropy', optimizer='adam',
              batch_size=128, nb_epoch=10, validation_split=0.2,
              shuffle=True):
    '''
    LSTM Model
    :param X: Train data input
    :param y: Train data label
    :param w2v_emb: Word embedding
    :param output_size: Output size
    :param dropout: Dropout ratio
    :param loss: Loss function
    :param optimizer: Optimizer
    :param batch_size: Batch size
    :param nb_epoch: Number of epoch
    :param validation_split: Training / Validation split
    :param shuffle: Shuffle training data before each epoch
    :return: LSTM model
    '''
    model = Sequential()
    model.add(w2v_emb)
    model.add(LSTM(64))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def CNNLSTM(X, y, w2v_emb, dropout=0.2,
            loss='binary_crossentropy', optimizer='adam',
            batch_size=128, nb_epoch=10, validation_split=0.2,
            shuffle=True):
    '''
    CNN + LSTM Model
    :param X: Train data input
    :param y: Train data label
    :param w2v_emb: Word embedding
    :param output_size: Output size
    :param dropout: Dropout ratio
    :param loss: Loss function
    :param optimizer: Optimizer
    :param batch_size: Batch size
    :param nb_epoch: Number of epoch
    :param validation_split: Training / Validation split
    :param shuffle: Shuffle training data before each epoch
    :return: CNN + LSTM model
    '''
    model = Sequential()
    model.add(w2v_emb)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def BiLSTM(X, y, w2v_emb, dropout=0.2,
           loss='binary_crossentropy', optimizer='adam',
           batch_size=128, nb_epoch=10, validation_split=0.2,
           shuffle=True):
    '''
    CNN + Bidirectional LSTM Model
    :param X: Train data input
    :param y: Train data label
    :param w2v_emb: Word embedding
    :param output_size: Output size
    :param dropout: Dropout ratio
    :param loss: Loss function
    :param optimizer: Optimizer
    :param batch_size: Batch size
    :param nb_epoch: Number of epoch
    :param validation_split: Training / Validation split
    :param shuffle: Shuffle training data before each epoch
    :return: CNN + Bidirectional model
    '''
    model = Sequential()
    model.add(w2v_emb)
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def Eval(X_test, y_test, name, model, verbose = True,
         output_path='model/'):
    '''
    Model evaluation
    :param X: Test data input
    :param y: Test data label
    :param model: Model
    :return: Accuracy and F1 scores
    '''
    print(f'{name} Model')
    y_pred = model.predict_classes(X_test)
    print(f'Test Accuracy:{accuracy_score(y_test, y_pred)}')
    print(f'Test F1:{f1_score(y_test, y_pred)}')
    if verbose:
        model.save(f'{output_path}{name}.h5')


def main(data_file, embed_path):

    # Load, classify and split data
    DF = LoadData(data_file, embed_path)
    DF.Load(max_len=200)

    # visualize the distribution of each class
    ax = DF.train_y.value_counts().plot(kind='bar', figsize=(14, 8),
                                        title="Number for Each Class (1 = high star, 0 = low star)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.show()


    # model training
    LSTM_model = base_LSTM(DF.train_X, DF.train_y, DF.w2v_emb, nb_epoch = 8)
    CNNLSTM_model = CNNLSTM(DF.train_X, DF.train_y, DF.w2v_emb, nb_epoch = 4)
    biLSTM_model = BiLSTM(DF.train_X, DF.train_y, DF.w2v_emb, nb_epoch = 4)

    # model performance on test data
    Eval(DF.test_X, DF.test_y, 'LSTM', LSTM_model, verbose=False)
    Eval(DF.test_X, DF.test_y, 'CNN + LSTM', CNNLSTM_model, verbose=False)
    Eval(DF.test_X, DF.test_y, 'CNN + Bidirectional LSTM', biLSTM_model, verbose=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        default="data/business_reviews2017.tsv",
                        help="2017 Yelp Business Reviews tsv file")

    parser.add_argument("--embed_path", type=str,
                        default="model/GoogleNews-vectors-negative300.bin",
                        help="Google News Pre-trained Word2Vec Model")

    args = parser.parse_args()

    main(args.data_file, args.embed_path)
