# Sentiment Classification of Yelp Business Reviews by Supervised Machine Learning and Deep Learning Approaches


## Data:
`yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` could be found [here](https://www.kaggle.com/yelp-dataset/yelp-dataset). 
We convert them from json file to tsv file, split data into pieces by year, filter business reviews published in 2017, and downsize to 40% of the data size as our data resource for this project.

Since is too large to upload on Github, please download the original data file in *data* directory before running the following scripts.

##### Data Processing
```{bash}
$ cat scripts/years.txt | xargs -I{} bash -c "bash scripts/split_json.sh {}"
$ ls data/*201*json | xargs -n 1 -I{} bash -c "python scripts/json2tsv.py --json={}"
```
`ImportReview.sql`, `created_yelp_database.sql` and `combine_business_review.py` will then be implemented to build our processed dataset `business_reviews2017.tsv` in *data* directory.
##### NOTE: 
To save the data downloading and converting time, the above steps can be skipped by using `business_reviews2017.tsv` uploaded in *data* directory, and all the following modeling and evaluation steps are using this file as well.

## Word2Vec
`GoogleNews-vectors-negative300.bin` could be found [here](https://code.google.com/archive/p/word2vec/).

Since is too large to upload on Github, please download the original file in *model* directory before testing the models.

## Machine Learning Models `Statsmodels.py`

Several supervised machine learning classifiers will be build on the training data (70%):
* Decision Tree
* Random Forests
* Naive Bayes
* Logistic Regression

Four features are chosen to represent test reviews:
* Bag of words
* Word TF-IDF
* N-gram Word level TF-IDF
* N-gram Character level TF-IDF

Example:

`python Statsmodels.py --data_file data/business_reviews2017.tsv --out_path data/business_reviews`

## Deep Learning Models `DLmodels.py`

We chosen common deep learning networks such LSTM and CNN to create three text classification architectures training on 70% of our data: 
* LSTM baseline model
* CNN and LSTM combined model
* Bidirectional LSTM and CNN combined model.

Example

`python DLmodels.py --data_file data/business_reviews2017.tsv --embed_path model/GoogleNews-vectors-negative300.bin`


## Model Performance on Test Dataset

All previous created models were evaluated on test set (30%), `Statsmodels.py` and `DLmodels.py` will print the accuracy and F1 scores on the screen.

##### Classical Supervised Machine Learning Models
![Supervised ML Models](stats_result.png?raw=true "ML models result")

##### Deep Learning Models
![DL Models](dl_result.png?raw=true "ML models result")
