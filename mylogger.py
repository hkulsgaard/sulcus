import logging
from logging.handlers import RotatingFileHandler

class mylogger(object):
    def __init__(self) -> None:
        pass

    def setup_logger(self, path):
        MAX_BYTES = 10000000 # Maximum size for a log file
        BACKUP_COUNT = 9 # Maximum number of old log files
        
        # The name should be unique, so you can get in in other places
        # by calling `logger = logging.getLogger('com.dvnguyen.logger.example')
        self.logger = logging.getLogger('logger.example') 
        self.logger.setLevel(logging.INFO) # the level should be the lowest level set in handlers

        log_format = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        info_handler = RotatingFileHandler(filename=path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
        info_handler.setFormatter(log_format)
        info_handler.setLevel(logging.INFO)
        self.logger.addHandler(info_handler)

    def print(self, text):
        self.logger.info(text)

    def get_log(self,path):
        logging.basicConfig(
        filename=path,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)

        log = logging.getLogger()
        print('called_get_log')
        
        return log
