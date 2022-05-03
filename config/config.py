import yaml
import sys
import os
import datetime

class Config():
    def __init__(self, production = True):
        if production is True:
            file_path = self.resource_path('config/config.yml') # window version
        else:
            file_path = self.resource_path('config/config_internal.yml') # window version
        with open(file_path, encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def get_data_path(self):
        data_path        = self.config['data_path']
        train_data_path  = data_path['train']
        test_data_path   = data_path['test']
        sample_data_path = data_path['sample']
        return train_data_path, test_data_path, sample_data_path
    

