import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the model-2
reg_model=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

#!st Page-3
@app.route('/')
def home():
    return render_template('home.html')

#2nd page-4
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

#run-5
if __name__ == "__main__":
    app.run(debug=True)