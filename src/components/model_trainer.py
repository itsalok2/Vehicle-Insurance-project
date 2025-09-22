import sys
from typing import Tuple
import numpy as np
import xgboost as xgb 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import *
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config

    def get_model_object_and_report(self,train:np.array,test:np.array)->Tuple[object,object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("training xgboost model")

            # Splitting the train and test data into features and target variables
            x_train,y_train,x_test,y_test=train[:,:-1],train[:,-1],test[:,:-1],test[:,-1]
            logging.info('train-test split done')

            # Initialize RandomForestClassifier with specified parameters
            model=xgb.XGBClassifier(
                max_depth = self.model_trainer_config.max_depth,
                learning_rate = self.model_trainer_config.learning_rate,
                gamma = self.model_trainer_config.gamma,
                subsample = self.model_trainer_config.subsample,
                colsample_bytree = self.model_trainer_config.colsample_bytree,
                min_child_weight = self.model_trainer_config.min_child_weight,
                reg_lambda = self.model_trainer_config.reg_lambda,   # L2 regularization
                reg_alpha = self.model_trainer_config.reg_alpha,     # L1 regularization
                n_estimators = self.model_trainer_config.n_estimators
            )

            # fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # prediction and evaluation metrics
            y_pred=model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            metric_artifact=ClassificationMetricArtifact(f1_score=f1,precision_score=precision,recall_score=recall)
            return model,metric_artifact
        except Exception as e:
            raise MyException(e,sys) from e
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info('train-test loaded')

            # Train model and get metrics
            trained_model,metric_artifact=self.get_model_object_and_report(train=train_arr,test=test_arr)
            logging.info('Model object and artifact loaded')

            # Load preprocessing object
            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info('Preprocessing obj loaded')

            # Check if the model's accuracy meets the expected threshold
            train_accuracy=accuracy_score(train_arr[:,-1],trained_model.predict(train_arr[:,:-1]))
            logging.info(f"Training accuracy: {train_accuracy:.2f} | Expected accuracy: {self.model_trainer_config.expected_accuracy:.2f}")
            if  train_accuracy < self.model_trainer_config.expected_accuracy:
                logging.info('No model found with score above then base score')
                raise Exception("No model found with score above the base score")
            
            # Save the final model object that includes both preprocessing and the trained model
            logging.info('Saving new model as performance is better than previous one.')
            my_model=MyModel(preprocessing_obj=preprocessing_obj,trained_model_obj=trained_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # create and return the ModelTrainerArtifact
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

            logging.info(f"model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e,sys) from e



