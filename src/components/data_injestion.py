import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass







##Initialising the data Injestion configuration
@dataclass
class DataInjestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')

class DataInjestion:
    def __init__(self) :
        self.injestion_config=DataInjestionConfig()
    
    def initate_data_injestion(self):
            
        logging.info("entered the data injestion method or component")

        try:
            df=pd.read_csv(r"Notebook/Data.csv")
            logging.info("Exported the dataset as dataframe")

            os.makedirs(os.path.dirname(self.injestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True)

            logging.info("Injestion of Data is completed")

            return(self.injestion_config.train_data_path,self.injestion_config.test_data_path)
            


        except Exception as e :
            logging.info("excepton occured at data injestion ")
            raise CustomException(e,sys)
            

