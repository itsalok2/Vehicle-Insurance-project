import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN,SCHEMA_FILE_APTH,CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import *
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import *

class DataTransformation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try: 
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            self._schemas_config= read_yaml_file(file_path=SCHEMA_FILE_APTH)

        except Exception as e:
            raise MyException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data_transformer_object(self)->Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # initialise tranformers
            numeric_transformer=StandardScaler()
            mm_scaler=MinMaxScaler()

            # load the schemas configuration
            num_features=self._schemas_config['num_features']
            mm_columns=self._schemas_config['mm_columns']
            logging.info(f"cols loaded from schema")

            # creating a preprocessor pipeline
            preprocessor=ColumnTransformer(
                transformers=[
                    ('StandaerScaler',numeric_transformer,num_features),
                    ('MinMaxScaler',mm_scaler,mm_columns)
                ],
                remainder='passthrough'
            )

            # Wrapping everything in a single pipeline
            final_pipeline=Pipeline(steps=[('Preprocessor',preprocessor)])
            logging.info(f"Final Pipeline Ready")
            logging.info(f"Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys)
        
    def map_gender_column(sef,df):
         """Map Gender column to 0 for Female and 1 for Male."""
         logging.info("Mapping 'gender' columns to binary values")
         df['Gender']=df['Gender'].map({'Female':0,'Male':1}).astype(int)
         return df
    
    def create_dummy_columns(self,df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("creating dummy variables for categorical features")
        df=pd.get_dummies(df,drop_first=True)
        return df
    
    def drop_id_column(self,df):
        """Drop the 'id' column if it exists."""
        logging.info('dropping "id" column')
        drop_col=self._schemas_config['drop_columns']
        if drop_col in df.columns:
            df=df.drop(drop_col,axis=1)
        return df
    
    def rename_columns(self,df):
        logging.info('renaming specific columns and casting to int')
        df=df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col]=df[col].astype('int')
        
        return df
    
    def initiate_date_transformation(self)->DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info('Data transformation started')
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df=self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df=self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info('Train test data loaded')

            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]

            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]

            # apply custom transformation in specified sequence
            input_feature_train_df=self.map_gender_column(input_feature_train_df)
            input_feature_train_df = self.drop_id_column(input_feature_train_df)
            input_feature_train_df = self.create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self.rename_columns(input_feature_train_df)

            input_feature_test_df = self.map_gender_column(input_feature_test_df)
            input_feature_test_df = self.drop_id_column(input_feature_test_df)
            input_feature_test_df = self.create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self.rename_columns(input_feature_test_df)
            logging.info('custom transformation applied to train and test data')

            # ✅ Verification step: check if 'id' column is still present
            if "id" in input_feature_train_df.columns:
                logging.error(f"❌ ID column still present in training data after preprocessing! Columns: {list(input_feature_train_df.columns)}")
                raise MyException("ID column was not dropped properly!", sys)
            else:
                logging.info(f"✅ Verified: ID column is not present. Final train columns: {list(input_feature_train_df.columns)}")

            logging.info('starting data transformation')
            preprocessor=self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt=SMOTEENN(sampling_strategy='minority')
            input_feature_train_final,target_feature_train_final=smt.fit_resample(
                input_feature_train_arr,target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr=np.c_[input_feature_train_final,np.array(target_feature_train_final)]
            test_arr=np.c_[input_feature_test_final,np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path
            )
        except Exception as e:
            raise MyException(e,sys) from e

    
    
    
