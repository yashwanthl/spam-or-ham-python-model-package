"""Spam - Spam Classification Production
Usage:
    spam-cli train <model-file>
    spam-cli predict <model-file> <emailtext>
    spam-cli (-h | --help)
Arguments:
    <model-file>   Serialized model file.
    <emailtext>     Text to be classified.
Options:
    -h --help                  Show this screen.
"""
import os

from docopt import docopt
import sys
from spam import Dataset, Model

def train_model(model_file):
    print('Instanciating model')
    dset = Dataset()

    print('Fetching data from data directory')
    emails = dset.get_data()
    print('Successfully fetched data. Here is the snapshot')
    print(emails.head())

    print('Instanciating model')
    model = Model()

    print('Training model')
    model.train(emails)
    print('Complete Training model')

    print('Storing model to ' + model_file)
    model.serialize(model_file)

def predict_model(model_file, email_text):
    print('Using '+ model_file +' predicting spam or not')

    print('Instanciating model')
    model = Model.deserialize(model_file)

    print('Predicting...')
    predict = model.predict([email_text])
    print(predict)
    if (predict[0] == 0):
        print('This text is NOT SPAM ')
    if (predict[0] == 1):
        print('This text is SPAM')


def main():
    arguments = docopt(__doc__)
    if arguments['train']:
        train_model(arguments['<model-file>'])
    elif arguments['predict']:
        predict_model(arguments['<model-file>'],
                arguments['<emailtext>'])

if __name__ == '__main__':
    main()
