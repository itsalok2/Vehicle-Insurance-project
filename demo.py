# from src.logger import logging

# logging.debug('this is debug message')

# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a=1+'z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e,sys) from e

#------------------running training pipeline only with dataingestion---------------------------------------
import logging
from src.pipline.training_pipeline import TrainPipeline
logging.getLogger("pymongo").setLevel(logging.WARNING)

pipeline=TrainPipeline()
pipeline.run_pipeline()

