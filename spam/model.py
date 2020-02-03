import pickle
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import time
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

class Model:
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
    
    def train(self, emails):
        # Check for duplicates and dop those rows
        emails.drop_duplicates(inplace=True)

        # Tokenization
        emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text))  

        # Removing stop words
        stop_words = set(nltk.corpus.stopwords.words('english'))
        emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words])

        # Removing 'Subject:'
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])

        # Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
        # Joining all tokens together in a string
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))

        # Removing apecial characters from each mail 
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

        #Lemmatization
        wnl = nltk.WordNetLemmatizer()
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))

        # Bag of Words
        self.vectorizer = CountVectorizer()
        counts = self.vectorizer.fit_transform(emails['filtered_text'].values)
        
        # Naive Bayes Classifier
        self.classifier = MultinomialNB()
        targets = emails['spam'].values
        self.classifier.fit(counts, targets)
    
    def predict(self, emailText):
        emailTexts_counts = self.vectorizer.transform(emailText)
        prediction = self.classifier.predict(emailTexts_counts)
        return prediction

    def serialize(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.classifier, f) 

    @staticmethod
    def deserialize(fname):
        model = Model()
        with open(fname, 'rb') as f:
            model.vectorizer = pickle.load(f)
            model.classifier = pickle.load(f)

            return model
