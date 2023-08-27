import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:

    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:

    def __init__(self):

        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        logging.info("splitting dependent and independent features from train and test data")
        try:
                
                X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                    "AdaBoost Classifier":AdaBoostClassifier()
                }



                model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)
                print(model_report)
                print('\n====================================================================================\n')
                logging.info(f'Model Report : {model_report}')

            
                ## To get best model score from dict
                best_model_score = max(sorted(model_report.values()))
                ## To get best model name from dict

                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]
                print(f'Best Model Found , Model Name : {best_model} , Accuracy Score : {best_model_score}')

                print('\n====================================================================================\n')
                logging.info(f'Best Model Found , Model Name : {best_model} , Accuracy Score : {best_model_score}')


                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
          

        
                if best_model_score<0.6:
                    raise CustomException("No best model found")
                else:
                    print(f'Best Model Found , Model Name : {best_model} , Accuracy Score : {best_model_score}')
                    print('\n====================================================================================\n')
                logging.info(f'Best Model Found , Model Name : {best_model} , Accuracy Score : {best_model_score}')
                logging.info(f"Best found model on both training and testing dataset")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model)
                logging.info("model saved")

                    
        except Exception as e:
            raise CustomException(e,sys)