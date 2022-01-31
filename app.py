from flask import Flask, request, url_for, redirect, render_template
import warnings
import pickle
warnings.filterwarnings("ignore")
import numpy as np

app = Flask(__name__)

model = pickle.load(open('RFC_model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("loan_status.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            int_features = [int(x) for x in request.form.values()]
            final = [np.array(int_features)]
            # print(int_features)
            # print(final)
            prediction = model.predict(final)

            if prediction == 0:
                return render_template('loan_status.html', pred='Loan Status - Approved [Yes!]')
            elif prediction == 1:
                return render_template('loan_status.html', pred='Loan Status - Rejected [No!]')
        except:
            return render_template('loan_status.html')
    else:
        return render_template('loan_status.html')    


if __name__ == '__main__':
    app.run(debug=True)