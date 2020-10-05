import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the logistic regression model
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Load the standardise data
filename2 = 'scaler.pkl'
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
    data_s = scaler.transform(data)
    prediction =classifier.predict_proba(data_s)[:,1]

    return round(prediction[0],2)


def main():

    st.sidebar.header('About')
    st.sidebar.info('This app is created to predict bank churners')
    st.sidebar.info('The dataset consist of 1000 records for France, Germany and Spain')

    from PIL  import Image
    image=Image.open('0_d58iZ6esNNcfntQ7.jpg')
    st.sidebar.image(image,width=300)

    st.title("Bank Churn Prediction App")

    html_temp = """
       <div style="background-color:#ADD8E6;padding:5px">
        <h2 style="color:white;text-align:center;">Web app Build using Streamlit, Deployed on Heroku </h2>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)

    CreditScore = st.number_input("Credit Score", min_value=0, max_value=10000, value=0)
    Geography = st.selectbox('Please select your country', ('France', 'Germany', 'Spain'))
    Gender = st.selectbox('Please select your Gender', ('Male','Female'))
    Age = st.number_input("Age (In Years)", min_value=0, max_value=100, value=0)
    Tenure = st.number_input("Tenure (In Years)", min_value=0, max_value=100, value=0)
    Balance = st.number_input("Balance", min_value=0, max_value=9999999999, value=0)
    NumOfProducts = st.number_input("Number Of Products used so far?", min_value=0, max_value=9999999999, value=0)
    HasCrCard = st.selectbox('Do you currently use credit card?', ('True', 'False'))
    IsActiveMember = st.selectbox('Are you active member?', ('True', 'False'))
    EstimatedSalary= st.number_input("Estimated Salary", min_value=0, max_value=9999999999, value=0)

    if st.button("Predict"):
        result = predict_prob(CreditScore,Geography,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Gender)
        print(result)

        if result>0.0 and result<=0.5:
             st.success("The Probability of Churn  is {}, The customer is expected to stay.".format(result))
        elif result>0.5 and result<=0.7:
             st.warning("The Probability of Churn is {}, There is Moderate risk of Customer being churned.".format(result))
        else:
             st.error("The Probability of Churn  is {}, There is High risk of Customer being churned.".format(result))


if __name__ == '__main__':
    main()