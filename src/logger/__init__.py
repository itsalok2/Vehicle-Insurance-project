import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

log_dir='logs'
log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
max_log_size=5*1024*1024
backup_count=3

log_dir_path=os.path.join(from_root(),log_dir)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir_path,log_file)

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """

    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter=logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

    filehandler=RotatingFileHandler(log_file_path,maxBytes=max_log_size,backupCount=backup_count)
    filehandler.setFormatter(formatter)
    filehandler.setLevel(logging.DEBUG)

    consolehandler=logging.StreamHandler()
    consolehandler.setFormatter(formatter)
    consolehandler.setLevel(logging.DEBUG)

    logger.addHandler(filehandler)
    logger.addHandler(consolehandler)


configure_logger()