import pandas as pd
import numpy as np
from os import path
import sys
import yaml

class configLoader():

	def __init__(self):
		self.config = dict()

    
	def load_from_path(self, csv_path, verbose=True):
		if not path.exists(csv_path): sys.exit('[ERROR] Config file does not exists')
		config_file = pd.read_csv(csv_path, sep='=', header=None)
		
		if verbose:
			print('[INFO]Experiment configuration loaded ({}):'.format(csv_path))

		for idx,row in config_file.iterrows():
			key = row[0].replace(' ','')

			# ignore row if starts with hashtag (comment)
			if not key[0] == '#':
				value = row[1].replace(' ','')
				if(value[0] == '['):
				# case array
					self.config[key] = self.load_array(value)
				
				else:
				# case single value
					self.config[key] = self.load_item(value)
			
				if verbose:
					print('	({}) = {}'.format(key,value))

		return self.config
	
	def load_item(self, str_item):
		# changes the data type of srt_item to the correct one

		if(str_item.isnumeric()):
		# case integer
			new_item = int(str_item)

		elif(str_item.replace('.','').isnumeric()):
		# case float
			new_item = float(str_item)

		else:
		# case string
			new_item = str_item

		return new_item
	
	def load_array(self, str_array):
		str_array = str_array.replace('[','').replace(']','')
		
		if(str_array.find('.')>-1):
			new_dtype = np.float64
		else:
			new_dtype = np.int32

		new_array = np.fromstring(str_array, sep=',', dtype=new_dtype)

		return new_array
	
#config = configLoader().load_from_path('./config/my_config.csv')
#print(type(config['n_epochs']))
#config_data = configparser.ConfigParser(converters={"val": lambda x: literal_eval(x)})
#config_data.read('./config/my_config.ini')
#data = config_data["training"]

#val = config_data.getval("training",'lr')

#print(type(data['lr']))
#print(type(literal_eval(data['lr'])))

with open('./config/my_config.yaml', 'r') as file:
   config = yaml.safe_load(file)

#print(type(config.get('lr')))
#print(type(config['lr']))
print(config['training']['lr'])