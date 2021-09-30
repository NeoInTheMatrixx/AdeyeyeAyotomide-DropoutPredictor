from flask import Flask, render_template,request
import pickle as pkl
import pandas as pd
import numpy as np
# from sklearn import tree


#Initialize the flask App
app = Flask(__name__)


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    model = pkl.load(open('modelrf.pkl', 'rb'))


    prediction = model.predict(final_features)
    prediction = " ".join(map(str,prediction))
    return render_template('output.html', data= prediction) 


if __name__ == "__main__":
    app.run(debug=True)
