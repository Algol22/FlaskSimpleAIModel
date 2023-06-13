from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        input_data = [int(request.form['feature1']), int(request.form['feature2'])]

        # Load the accounting data
        accounting_data = pd.read_excel('accounting.xlsx')
        X = accounting_data.drop(columns=['DESC'])
        y = accounting_data['DESC']

        # Train the model
        model = DecisionTreeClassifier()
        model.fit(X, y)

        # Make prediction
        prediction = model.predict([input_data])

        # Render the index.html template with the prediction result
        return render_template('index.html', prediction=prediction[0], feature1=input_data[0], feature2=input_data[1])

    # Render the input form template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
