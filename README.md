# Sentiment Classification of Yelp Business Reviews by Supervised Machine Learning and Deep Learning Approaches


## Data:
`yelp_academic\_dataset\_review.json` could be found [here](https://www.yelp.com/dataset). 
We convert it from json file to tsv file, split data into pieces by year, and filter business reviews published in 2017 as our data resource for this project.

Since is too large to upload on Github, please download the original data file in *data* directory before running the following scripts.
```{bash}
$ ls ./data/*201*json | xargs -n 1 -I{} bash -c "python json2tsv.py --json={}"
$ cat scripts/years.txt | xargs -I{} bash -c "bash scripts/split_json.sh {}"
```
The above step can be skipped simply by using the processed `business_reviews2017.tsv` in *data* directory.

## Word2Vec
`GoogleNews-vectors-negative300.bin` could be found [here](https://code.google.com/archive/p/word2vec/).

Since is too large to upload on Github, please download the original file in *model* directory before testing the models.

## Deep Learning Models

LSTM model, CNN and LSTM combined model, CNN and Bidirectional LSTM model.

* Example

`python DLmodels.py --data_file ../data/business_reviews2017.tsv --out_path ../data/business_reviews` 


## Model Performance on Test Dataset

* Naive Bayes Model - WordCount:

Accuracy: 0.756

F1: 0.746

* Naive Bayes Model - Word TF-IDF:

Accuracy:0.754

F1:0.726

* Naive Bayes Model - Ngram TF-IDF:

Accuracy:0.736

F1:0.710

* Naive Bayes Model - Ngram CharTF-IDF:

Accuracy:0.736

F1:0.688

* Logistic Regression - WordCount:

Accuracy:0.759

F1:0.733

* Logistic Regression - Word TF-IDF:

Accuracy:0.773

F1:0.751

* Logistic Regression - Ngram TF-IDF:

Accuracy:0.750

F1:0.727

* Logistic Regression - Ngram CharTF-IDF:

Accuracy:0.766

F1:0.742

* LSTM Model

Accuracy:0.624

F1:0.513

* CNN + LSTM Model

Accuracy:0.715

F1:0.674

* CNN + Bidirectional LSTM Model

Accuracy:0.720

F1:0.685
