# import libraries
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
import pickle

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    """
    Load the data from the database 'database_filepath' and 
    split it into feature 'X' and targets 'y'
    
    # Parameters
       database_filepath (str): full path to database
    
    # Returns
        X (dataframe): Feature data, just the messages
        y (dataframe): Classification labels
        category_names (list): List of the category names for classification

    """    
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text function
    
    # Parameters
    text (str): Text message to be tokenized

    # Returns
    clean_tokens (list): List of tokens extracted from text

    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Extract word tokens from the provided text
    tokens = word_tokenize(text)
    
    #Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    
    return clean_tokens


# Custom transformer to extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Refer to Udacity ML Pipeline Notebook
    
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def build_model():
    """
    Return Grid Search model with pipeline and Classifier
    
    # Returns
    model (scikit-learn GridSearchCV): Grid search model object
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression()))

    ])
    
    parameters = {  
        'features__text_pipeline__tfidf__smooth_idf': (True, False),
        'clf__estimator__penalty' : ['l1', 'l2']
    }    

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model    


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Model Evaluation Function. 
   
    # Parameters
    model (GridSearchCV or Scikit-learn Pipeline Object) : Trained ML Model
    X_test (DataFrame) : Test Features
    Y_test (DataFrame) : Test Labels
    category_names (List): Category Name List
    
    # Output
        accuracy, f1score
    
    """

    # predict on test data
    y_pred = model.predict(X_test)
    
    metrics = []
    for i in range(len(category_names)):
        accuracy = category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])       
        metrics.append(accuracy)
    

    print(pd.DataFrame(metrics, columns=['Category', 'accuracy']).set_index('Category'))
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Dumps the model to a given filepath
    
    # Parameters
        model (scikit-learn model): The fitted model
        model_filepath (str): the filepath to save the model 

    # Returns 
        None
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()