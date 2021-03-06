import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn import linear_model, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import random

random.seed(0)




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


## Model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y,f):
    """
    fit the training dataset on the classifier
    """
    
    classifier.fit(feature_vector_train, label)

    # predictions
    predictions = classifier.predict(feature_vector_valid)

    print(f'Test Accuracy:{accuracy_score(predictions, valid_y)}',file=f)
    print(f'Test F1:{f1_score(predictions, valid_y)}',file=f)


def main(data_file, out_path):
    
    random.seed(0)

    # Load, classify and split data
    DF = LoadData(data_file, out_path, verbose=False)

#    # visualize the distribution of each class
#    ax = DF.data['class'].value_counts().plot(kind='bar',figsize=(14,8),
#                title="Number for Each Class (1 = high star, 0 = low star)")
#    ax.set_xlabel("Class")
#    ax.set_ylabel("Count")
#    plt.show()

    ## features
    
    # 1. use word counts as features counter
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=3000)
    count_vect.fit(DF.data['text'])

    # use word counter to transform train_set and valid set
    xtrain_count = count_vect.transform(DF.train['text'])
    xvalid_count = count_vect.transform(DF.test['text'])

    # 2. use TF-IDF as feature sets

    # word-level TF-IDF
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=3000)
    tfidf_vect.fit(DF.data['text'])
    xtrain_tfidf = tfidf_vect.transform(DF.train['text'])
    xvalid_tfidf = tfidf_vect.transform(DF.test['text'])

    # ngram-level TF-IDF
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                       ngram_range=(2, 3), max_features=3000)
    tfidf_vect_ngram = tfidf_vect_ngram.fit(DF.data['text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(DF.train['text'])
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(DF.test['text'])

    # char-level TF-IDF
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                             max_features=3000)
    tfidf_vect_ngram_chars.fit(DF.data['text'])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(DF.train['text'])
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(DF.test['text'])



    with open('result.txt','w') as f:  
        
    # ----------------------- Baseline: Decision Tree  ----------------------
       
    ## DT - on word count vector
        print("\nDT - WordCount:",file=f)
        train_model(tree.DecisionTreeClassifier(), xtrain_count, DF.train['class'], xvalid_count,
                    DF.test['class'],f)
    
        ## DT - on word-level TF-IDF
        print("\nDT - WordTF-IDF:",file=f)
        train_model(tree.DecisionTreeClassifier(), xtrain_tfidf, DF.train['class'], xvalid_tfidf,
                    DF.test['class'],f)
    
        ## DT - on ngram-level TF-IDF
        print("\nDT - NgramTF-IDF:",file=f)
        train_model(tree.DecisionTreeClassifier(), xtrain_tfidf_ngram, DF.train['class'],
                               xvalid_tfidf_ngram, DF.test['class'],f)
    
        ## DT - on ngram-char-level TF-IDF
        print("\nDT - NgramCharTF-IDF:",file=f)
        train_model(tree.DecisionTreeClassifier(), xtrain_tfidf_ngram_chars, DF.train['class'],
                               xvalid_tfidf_ngram_chars, DF.test['class'],f)
   

    # ----------------------- Naive Bayes ---------------------------

        
        ## Naive Bayes - on word count vector
        print("\nNB - WordCount:",file=f)
        train_model(naive_bayes.MultinomialNB(), xtrain_count, DF.train['class'], xvalid_count, DF.test['class'],f)
    
        ## Naive Bayes - on word-level TF-IDF
        print("\nNB - WordTF-IDF:",file=f)
        train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, DF.train['class'], xvalid_tfidf, DF.test['class'],f)

        ## Naive Bayes - on ngram-level TF-IDF
        print("\nNB - NgramTF-IDF:",file=f)
        train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, DF.train['class'], xvalid_tfidf_ngram,
                    DF.test['class'],f)
    
        ## Naive Bayes - on ngram-char-level TF-IDF
        print("\nNB - NgramCharTF-IDF:",file=f)
        train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, DF.train['class'],
                               xvalid_tfidf_ngram_chars, DF.test['class'],f)

        # ----------------------- LogisticRegression ---------------------------
    
        ## LR - on word count vector
        print("\nLR - WordCount:",file=f)
        train_model(linear_model.LogisticRegression(), xtrain_count, DF.train['class'], xvalid_count,
                    DF.test['class'],f)
    
        ## LR - on word-level TF-IDF
        print("\nLR - WordTF-IDF:",file=f)
        train_model(linear_model.LogisticRegression(), xtrain_tfidf, DF.train['class'], xvalid_tfidf,
                    DF.test['class'],f)
    
        ## LR - on ngram-level TF-IDF
        print("\nLR - NgramTF-IDF:",file=f)
        train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, DF.train['class'],
                               xvalid_tfidf_ngram, DF.test['class'],f)
    
        ## LR - on ngram-char-level TF-IDF
        print("\nLR - NgramCharTF-IDF:",file=f)
        train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, DF.train['class'],
                               xvalid_tfidf_ngram_chars, DF.test['class'],f)
    
    
        # ----------------------- Random Forest  ---------------------------
        
        ## RF - on word count vector
        print("\nRF - WordCount:",file=f)
        train_model(RandomForestClassifier(), xtrain_count, DF.train['class'], xvalid_count,
                    DF.test['class'],f)
    
        ## RF - on word-level TF-IDF
        print("\nRF - WordTF-IDF:",file=f)
        train_model(RandomForestClassifier(), xtrain_tfidf, DF.train['class'], xvalid_tfidf,
                    DF.test['class'],f)
    
        ## RF - on ngram-level TF-IDF
        print("\nRF - NgramTF-IDF:",file=f)
        train_model(RandomForestClassifier(), xtrain_tfidf_ngram, DF.train['class'],
                               xvalid_tfidf_ngram, DF.test['class'],f)
    
        ## RF - on ngram-char-level TF-IDF
        print("\nRF - NgramCharTF-IDF:",file=f)
        train_model(RandomForestClassifier(), xtrain_tfidf_ngram_chars, DF.train['class'],
                               xvalid_tfidf_ngram_chars, DF.test['class'],f)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        default="data/business_reviews2017.tsv",
                        help="2017 Yelp Business Reviews tsv file")
    parser.add_argument("--out_path", type=str,
                        default="data/business_reviews",
                        help="Dir to write train/test data")

    args = parser.parse_args()

    main(args.data_file, args.out_path)
