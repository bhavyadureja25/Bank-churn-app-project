# -*- coding: utf-8 -*-
"""
@author: @bhavyadureja
"""

# -*- coding: utf-8 -*-
"""Created by @Bhavyadureja"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Load the logistic regression model
filename = 'model.pkl'
filename2= 'scaler.pkl'
classifier = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open(filename2, 'rb'))
def predict_prob(CreditScore,Geography,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Gender):

    if Gender == 'Male':
        Gender_Male = 1
    else:
        Gender_Male = 0

    if HasCrCard=='Yes':
        HasCrCard=1
    else:
        HasCrCard=0

    if IsActiveMember=='Yes':
        IsActiveMember=1
    else:
        IsActiveMember=0

    if Geography=="France":
        Geography=0
    elif Geography=="Spain":
        Geography=1
    else:
        Geography=2

    try:
        age_to_tenure= Age/Tenure
        BalanceToSalaryRatio= EstimatedSalary/Balance
        ScoreToBalance= CreditScore/Balance
        SalaryToAge= Age/EstimatedSalary
        ProductsToBalance= NumOfProducts/Balance
    except:
        age_to_tenure= 0
        BalanceToSalaryRatio= 0
        ScoreToBalance= 0
        SalaryToAge= 0
        ProductsToBalance= 0

    data = np.array([[CreditScore,Geography,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Gender_Male,age_to_tenure,BalanceToSalaryRatio,ScoreToBalance,SalaryToAge,ProductsToBalance]])
    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)
    prediction =classifier.predict_proba(data_s)[:,1]

    return round(prediction[0],5)

def main():
    "App"
    st.title("Bank Churn Prediction App")
    html_temp = """
           <div style="background-color:#ADD8E6;padding:5px">
            <h2 style="color:white;text-align:center;">Web app Build using Streamlit, Deployed on Heroku </h2>
           </div>
           """
    menu = ('Information', 'Bank Churn Prediction')
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Information':
        st.header('Information')
        st.subheader('What is Bank Churn Prediction?')
        st.text('The Bank Churn is the prediction the movement of customers '
                'from one company to another')
        st.subheader('What does Bank Churn Prediction App do?')
        st.text('Bank Churn Prediction App helps you to predict the churning rate of your company.')

    else:
        st.text('Fill in the required details below:')
        CreditScore = st.number_input("Credit Score", min_value=0, max_value=10000, value=0)
        Geography = st.selectbox('Please select your country', ('France', 'Germany', 'Spain'))
        Gender = st.selectbox('Please select your Gender', ('Male', 'Female'))
        Age = st.number_input("Age (In Years)", min_value=0, max_value=100, value=0)
        Tenure = st.number_input("Tenure (In Years)", min_value=0, max_value=100, value=0)
        Balance = st.number_input("Balance", min_value=0, max_value=9999999999, value=0)
        NumOfProducts = st.number_input("Number Of Products used so far?", min_value=0, max_value=9999999999, value=0)
        HasCrCard = st.selectbox('Do you currently use credit card?', ('True', 'False'))
        IsActiveMember = st.selectbox('Are you active member?', ('True', 'False'))
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0, max_value=9999999999, value=0)

        if st.button("Predict"):
            result = predict_prob(CreditScore, Geography, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
                              EstimatedSalary, Gender)
            print(result)

            if result > 0.0 and result <= 0.5:
                st.success("The Probability of Churn  is {}, The customer is expected to stay.".format(result))
            elif result > 0.5 and result <= 0.7:
                st.warning("The Probability of Churn is {}, There is Moderate risk of Customer being churned.".format(result))
            else:
                st.error("The Probability of Churn  is {}, There is High risk of Customer being churned.".format(result))

if __name__== '__main__':
    main()