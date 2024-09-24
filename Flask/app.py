import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))




@app.route('/')#route to display the home page
def home():
    return render_template('index.html')

@app.route('/about')
def aboutpage():
    
    return render_template('about.html')

@app.route('/info')
def infopage():
    
    return render_template('info.html')

@app.route('/details')
def deatilspage():
    
    return render_template('details.html')


@app.route('/predict', methods=["POST"])
def predict():
    # Collect input data
    Gender = (request.form["Gender"])
    Age = (request.form["Age"])
    History = (request.form["History"])
    Patient = (request.form["Patient"])
    TakeMedication = (request.form["TakeMedication"])
    Severity = (request.form["Severity"])
    BreathShortness = (request.form["BreathShortness"])
    VisualChanges = (request.form["VisualChanges"])
    NoseBleeding = (request.form["NoseBleeding"])
    Whendiagnoused = (request.form["Whendiagnoused"])
    Systolic = (request.form["Systolic"])
    Diastolic = (request.form["Diastolic"])
    ControlledDiet = (request.form["ControlledDiet"])
    
    
    features_values=np.array([[Gender,Age,History,Patient,TakeMedication,Severity,BreathShortness,VisualChanges,
                               NoseBleeding, Whendiagnoused,Systolic,Diastolic,ControlledDiet]])
    
    df = pd.DataFrame(features_values, columns = ['Gender','Age','History','Patient','TakeMedication','Severity','BreathShortness','VisualChanges',
                                                  'NoseBleeding','Whendiagnoused','Systolic','Diastolic','ControlledDiet'])
    
      # Convert 'gender' to numerical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Age'] = df['Age'].map({'18-34': 0, '35-50': 1, '51-64': 2, '65+': 3})
    df['History'] = df['History'].map({'No': 0, 'Yes': 1})
    df['Patient'] = df['Patient'].map({'No': 0, 'Yes': 1})
    df['TakeMedication'] = df['TakeMedication'].map({'No': 0, 'Yes': 1})
    df['Severity'] = df['Severity'].map({'Mild': 0, 'Moderate': 1, 'Server': 2})
    df['BreathShortness'] = df['BreathShortness'].map({'No': 0, 'Yes': 1})
    df['VisualChanges'] = df['VisualChanges'].map({'No': 0, 'Yes': 1})
    df['NoseBleeding'] = df['NoseBleeding'].map({'No': 0, 'Yes': 1})
    df['Whendiagnoused'] = df['Whendiagnoused'].map({'<1 Year': 1, '1 - 5 Years': 0, '>5 Years': 2})
    df['Systolic'] = df['Systolic'].map({'111 - 120': 1, '121 - 130': 2, '130+': 3, '100+': 0})
    df['Diastolic'] = df['Diastolic'].map({'70 - 80': 2 ,'81 - 90': 3, '91 - 100': 4, '100+': 0, '130+': 1})
    df['ControlledDiet'] = df['ControlledDiet'].map({'No': 0, 'Yes': 1})

    
    
    # Ensure all columns are numerical
    df = df.astype(float)
    
    prediction = model.predict(df)
    print(prediction[0])
    
    if prediction[0] == 0:
        result="NORMAL"
    elif prediction[0] == 1:
        result="HYPERTENSION (Stage-1)"
    elif prediction[0] == 2:
        result="HYPERTENSION (Stage-2)"
    else:
        result="HYPERTENSIVE CRISIS"
    print(result)
    
    
    return render_template('prediction.html', prediction_text=result)



if __name__ == "__main__":
        app.run(debug = True)