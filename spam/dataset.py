import pandas as pd

class Dataset:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
    
    def get_data(self):
        file = self.data_dir + '/emails.csv'
        return pd.read_csv(file)
