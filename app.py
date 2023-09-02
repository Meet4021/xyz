# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:34:30 2020

@author: Rohith
"""

import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score

model=pickle.load(open('bank_rf.pkl', 'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    df=pd.read_csv("C:\Users\dhruv\OneDrive\Desktop\Final\bank-additional-full.csv")
    bank.head()
    bank_data.head()
    from sklearn.model_selection import train_test_split
    X=scaled_data.drop(['pdays','month','cons.conf.idx','loan','housing','y'],axis=1)
    df['Target'] = df['Target'].map({'no': 0, 'yes': 1})
    y=scaled_data.y
   
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    
    if request.method == 'POST':
        my_prediction = rf.predict(X_test)
        output = round(my_prediction[0], 2)
        return render_template('index.html',prediction = output)

if __name__ == '__main__':
    app.run(debug=True)



