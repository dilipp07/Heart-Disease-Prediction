from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import os
import sys

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:

        data=CustomData(
            Age=int(request.form.get('Age')),
            Sex = int(request.form.get('Sex')),
            Chest_pain_type = int(request.form.get('Chest_pain_type')),
            BP = int(request.form.get('BP')),
            
            Cholesterol =int (request.form.get('Cholesterol')),
            FBS_over_120 =int (request.form.get('FBS_over_120')),
            EKG_results =int (request.form.get('EKG_results')),
            Max_HR =int (request.form.get('Max_HR')),
            Exercise_angina =int (request.form.get('Exercise_angina')),
            ST_depression =float (request.form.get('ST_depression')),
            Slope_of_ST =int (request.form.get('Slope_of_ST')),
            Number_of_vessels_fluro =int (request.form.get('Number_of_vessels_fluro')),
            Thallium =int (request.form.get('Thallium')),


        )
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)