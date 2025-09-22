import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from src.exception import MyException
from src.logger import logging

class TragetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
        
class MyModel:
    def __init__(self,preprocessing_obj:Pipeline,trained_model_obj: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_obj = preprocessing_obj
        self.trained_model_obj = trained_model_obj

    def predict(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info('Starting prediction process')

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object

            transformed_fea=self.preprocessing_obj.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            prediction=self.trained_model_obj.predict(transformed_fea)
            
            return prediction
        
        except Exception as e:
            logging.error('Error occured in predict method',exc_info=True)
            raise MyException(e,sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_obj).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_obj).__name__}()"


    