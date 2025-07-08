from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['Sex'] = 0 if df['Sex'][0] == 'male' else 1
    df['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}[df['Embarked'][0]]
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    input_data = {
        'Pclass': int(form_data['Pclass']),
        'Sex': form_data['Sex'],
        'Age': float(form_data['Age']),
        'SibSp': int(form_data['SibSp']),
        'Parch': int(form_data['Parch']),
        'Fare': float(form_data['Fare']),
        'Embarked': form_data['Embarked']
    }

    df = preprocess_input(input_data)
    prediction = model.predict(df)[0]
    result = '✅ Survived' if prediction == 1 else '❌ Did not survive'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
