import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import projectFtencoder
import utils
import yaml
import sys
import os

print('\n--------------------------')
print('DIAGONAL SULCUS DETECTION')
print('--------------------------')

# Load configuration file
try:
	config_path = sys.argv[1]

except:
	config_path = utils.askConfigFile(os.getcwd(), title='Select the YAML configuration file')

if os.path.exists(config_path):
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)
	print('[INFO]Loaded configuration file:"{}"'.format(config_path))
	
	#Run prediction
	pFTE = projectFtencoder.projectFtencoder()
	pFTE.run_predict(config)
else:
	print('[ERROR]Configuration file does not exist "{}"'.format(config_path))

print('\n')