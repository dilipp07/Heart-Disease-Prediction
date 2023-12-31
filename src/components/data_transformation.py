import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler



from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
        
            numerical_cols = ["Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120","EKG results","Max HR",
            "Exercise angina","ST depression","Slope of ST","Number of vessels fluro","Thallium"]


            
            logging.info('Pipeline Initiated')
            


            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])


            

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols)])
                        
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)



            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Heart Disease'
            target_feature_train_sr=train_df[target_column_name]
            target_feature_train_df=target_feature_train_sr.to_frame()
            target_feature_test_df=test_df[target_column_name]
            target_feature_test_df=target_feature_test_df.to_frame()

            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_train_arr=input_feature_train_arr.toarray()
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            # input_feature_test_arr=input_feature_test_arr.toarray()
            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]





            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)