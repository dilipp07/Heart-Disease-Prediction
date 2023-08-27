
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        model_list = []
        model_report = {}  # Initialize the model report dictionary
        Accuracy = []
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            Train_result, accuracy_train = model_eval(y_train, y_train_pred)
            Test_result, accuracy_test = model_eval(y_test, y_test_pred)

            model_name = list(models.keys())[i]
            model_list.append(model_name)
            model_report[model_name] = accuracy_test  # Store accuracy test score in the dictionary
            
            # print(model_name)
            # print('Model performance for Training set')
            # print(Train_result)
            # print('----------------------------------')
            # print('Model performance for Test set')
            # print(Test_result)
            # Accuracy.append(accuracy_test)
            # print('=' * 35)
            # print('\n')

        return model_report  # Return the model list and model report


    except Exception as e:
        logging.info('Exception occured during model evaluation')
        raise CustomException(e,sys)



def model_eval(y_true, y_pred):
    try:

        tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        specificity = tn/(fp+tn)
        F1_score = 2*recall*precision/(recall+precision)
        auc = roc_auc_score(y_true,y_pred)
        result = {   "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "F1 Score": F1_score,
                    'auc score':auc}
        return result , accuracy
        

        



    except Exception as e:
        logging.info('Exception occured during model evaluation')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
        