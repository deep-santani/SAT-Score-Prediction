# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:31:22 2020

@author: deep
"""

from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        SAT = int(request.form['SAT'])
        rand = int(request.form['Rand 1,2,3'])
        
        my_prediction = model.predict([[SAT,rand]])
        
        return render_template('result.html',prediction_text="Your GPA Score is {}".format(my_prediction))

if __name__ == '__main__':
	app.run(debug=True)
