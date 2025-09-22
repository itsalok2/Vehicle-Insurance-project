import os 
import datetime

DATABASE_NAME='vehicle_insurance_project'
COLLECTION_NAME='vehicle_data'
MONGODB_URL_KEY='MONGODB_URL'

PIPELINE_NAME=''
ARTIFACT_DIR='artifact'

MODEL_FILE_NAME='model.pkl'

TARGET_COLUMN='Response'
CURRENT_YEAR=datetime.date.today().year
PREPROCESSING_OBJECT_FILE_NAME='preprocessing.pkl'

FILE_NAME='data.csv'
TRAIN_FILE_NAME='train.csv'
TEST_FILE_NAME='test.csv'
SCHEMA_FILE_APTH=os.path.join('config','schema.yaml')

# WRITE YOUR AWS 
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str='vehicle_data'
DATA_INGESTION_DIR_NAME: str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str='feature_store'
DATA_INGESTION_INGESTED_DIR: str='ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float=0.25


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str="data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str= 'report.yaml'

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str='data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str='transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str='transformed_object'

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_MAX_DEPTH: int = 9
MODEL_TRAINER_LEARNING_RATE: float = 0.03482299073723265
MODEL_TRAINER_GAMMA: float = 0.45343123695179915
MODEL_TRAINER_SUBSAMPLE: float = 0.8481052686304679
MODEL_TRAINER_COLSAMPLE_BYTREE: float = 0.621632066242344
MODEL_TRAINER_MIN_CHILD_WEIGHT: int = 2
MODEL_TRAINER_LAMBDA: float = 3.515017261833032e-05   # L2 regularization
MODEL_TRAINER_ALPHA: float = 0.0008594224714468341    # L1 regularization
MODEL_TRAINER_N_ESTIMATOR: int = 870

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "vehicle-proj-mlops"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000