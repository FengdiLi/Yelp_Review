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
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Bidirectional
from sklearn.metrics import accuracy_score, f1_score
# from keras.models import load_model


class LoadData:
    '''
    Load, classify and split data
    '''
    def __init__(self, data_file, out_path, verbose=True):
        self.data = pd.read_csv(data_file, sep = '\t', index_col = 0)
        # assign review samples to two classes using [0,4) and [4, 5] criteria
        self.data['class'] = (self.data['stars'] >= 4).astype(int)
        self.data = self.data[['text', 'class']]
        self.data['text'] = self.data['text'].apply(CleanText)
        np.random.seed(1)
        self.train, self.test = train_test_split(self.data, train_size=0.7)
        # optional file saving
        if verbose:
            self.data.to_csv(out_path + '.tsv', sep='\t', index=False)
            self.train.to_csv(out_path+'_train.tsv', sep='\t', index=False)
            self.test.to_csv(out_path+'_test.tsv', sep='\t', index=False)


def CleanText(string):
    '''
    String cleaning
    :param string:
    :return: Cleaned review text
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


def Padding(data, max_len = 50):
    '''
    Padding vector to specified length
    :param data: Text input
    :param max_len: Padding length
    :return: Padded vector
    '''
    return pad_sequences(data, padding='post', truncating='post', maxlen = max_len)


def WordEmbedding(X, y, embed_path='../model/GoogleNews-vectors-negative300.bin',
                  max_features = 3000, w2v_size = 300, max_len = 50):
    '''
    Create word embedding
    :param X: train data input
    :param y: train data label
    :param embed_path: Path to pre-trained word2vec model
    :param max_features: Maximum number of features
    :param w2v_size: Word2vec size
    :param max_len: text padding length
    :return: Processed training data input and label, tokenizer, word embedding
    '''
    # tokenization & vectorizization
    tk = text.Tokenizer(num_words=max_features, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n',
                        split=" ")
    tk.fit_on_texts(X)
    word_index = tk.word_index
    # padding
    X_train = Padding(tk.texts_to_sequences(X), max_len)
    y_train = y
    # load google news pre-trained model
    w2v_model = KeyedVectors.load_word2vec_format(embed_path, binary=True)
    # Create word embedding vector matrix using pre-trained model
    w2v_matrix = np.zeros((len(word_index) + 1, w2v_size))
    for word,i in word_index.items():
        if word in w2v_model.vocab:
            w2v_matrix[i] = w2v_model[word]
    w2v_emb = Embedding(len(word_index)+1, w2v_size, weights=[w2v_matrix],
                            input_length=max_len)
    return X_train, y_train, tk, w2v_emb


def base_LSTM(X, y, w2v_emb, output_size=100, dropout=0.2,
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
    model.add(LSTM(output_size))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.layers[1].trainable = False
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def CNNLSTM(X, y, w2v_emb, output_size=100, dropout=0.2,
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
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.layers[1].trainable = False
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def BiLSTM(X, y, w2v_emb, output_size=100, dropout=0.2,
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
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(output_size)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.layers[1].trainable = False
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size,
              epochs=nb_epoch, validation_split=validation_split,
              shuffle=shuffle)

    return model


def Eval(X, y, name, model, verbose = True,
         output_path='../model/'):
    '''
    Model evaluation
    :param X: Test data input
    :param y: Test data label
    :param model: Model
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


def main(data_file, out_path):

    # Load, classify and split data
    DF = LoadData(data_file, out_path, verbose=False)

    # visualize the distribution of each class
    ax = DF.data['class'].value_counts().plot(kind='bar',figsize=(14,8),
                title="Number for Each Class (1 = high star, 0 = low star)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.show()

    # data pre-processing
    X_train, y_train, tk, w2v_emb = WordEmbedding(DF.train['text'], DF.train['class'], max_len = 100)

    # model training
    LSTM_model = base_LSTM(X_train, y_train, w2v_emb, output_size = 64)
    CNNLSTM_model = CNNLSTM(X_train, y_train, w2v_emb, output_size = 64, nb_epoch = 3)
    biLSTM_model = BiLSTM(X_train, y_train, w2v_emb, output_size = 64, nb_epoch=3)

    # model performance on test data
    X_test = Padding(tk.texts_to_sequences(DF.test['text']), max_len = 100)
    Eval(X_test, DF.test['class'], 'LSTM', LSTM_model, verbose=False)
    Eval(X_test, DF.test['class'], 'CNN + LSTM', CNNLSTM_model, verbose=False)
    Eval(X_test, DF.test['class'], 'CNN + Bidirectional LSTM', biLSTM_model, verbose=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        default="../data/business_reviews2017.tsv",
                        help="2017 Yelp Business Reviews tsv file")
    parser.add_argument("--out_path", type=str,
                        default="../data/business_reviews",
                        help="Dir to write train/test data")

    args = parser.parse_args()

    main(args.data_file, args.out_path)
