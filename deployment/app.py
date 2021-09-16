from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template
import requests

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Main page
    return render_template('index.html')


def model_predict():
    ##less class

   
    
    
    return 'less class'



def model_predict2():
    ##more class 

    
    return 'more class'
    

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = model_predict()
        return result
    
    return None

@app.route('/predict2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = model_predict2()
        return result
    
    return None

if __name__ == "__main__":
    app.run(debug=True)