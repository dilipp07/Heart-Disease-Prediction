import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info("done prediction")
            return pred
            logging.info("returned pred value")

            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)





class CustomData:
    def __init__(self,
    Age:int,
    Sex:int,
    Chest_pain_type:int,
    BP:int,
    Cholesterol:int,
    FBS_over_120 :int,
    EKG_results:int,
    Max_HR:int,
    Exercise_angina:int,
    ST_depression:float,
    Slope_of_ST:int,
    Number_of_vessels_fluro :int,
    Thallium :int):
    
        
        self.Age=Age
        self.Sex=Sex
        self.Chest_pain_type=Chest_pain_type
        self.BP=BP
        self.Cholesterol=Cholesterol
        self.FBS_over_120=FBS_over_120
        self.EKG_results=EKG_results
        self.Max_HR=Max_HR
        self.Exercise_angina=Exercise_angina
        self.ST_depression=ST_depression
        self.Slope_of_ST=Slope_of_ST
        self.Number_of_vessels_fluro=Number_of_vessels_fluro
        self.Thallium=Thallium
        
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Sex':[self.Sex],
                'Chest pain type':[self.Chest_pain_type],
                'BP':[self.BP],
                'Cholesterol':[self.Cholesterol],
                'FBS over 120':[self.FBS_over_120],
                'EKG results':[self.EKG_results],
                'Max HR':[self.Max_HR],
                'Exercise angina':[self.Exercise_angina],
                'ST depression':[self.ST_depression],
                'Slope of ST':[self.Slope_of_ST],
                'Number of vessels fluro':[self.Number_of_vessels_fluro],
                'Thallium':[self.Thallium]




                }

            df = pd.DataFrame(custom_data_input_dict)
           
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)