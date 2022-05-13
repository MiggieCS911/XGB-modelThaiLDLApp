#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#install library
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

#load model
@st.cache(ttl=60*60*12) #ttl -> time limit cache (second), max_entries -> number limit
def load_xgbDLP3v():
    xgb3v = xgb.XGBRegressor()
    xgb3v.load_model('XGBmodel_DLP.json')
    return xgb3v
# xgbDLP3v = xgb.XGBRegressor()
# xgbDLP3v.load_model('XGBmodel_DLP.json')
@st.cache(ttl=60*60*12)
def load_xgbDLP2v():
    xgb2v = xgb.XGBRegressor()
    xgb2v.load_model('XGBmodel_DLP2V.json')
    return xgb2v
# xgbDLP2v = xgb.XGBRegressor()
# xgbDLP2v.load_model('XGBmodel_DLP2V.json')

xgbDLP3v = load_xgbDLP3v()
xgbDLP2v = load_xgbDLP2v()

#predict LDL function
#3variables: Chol,HDL, TG
def fit_xgbDLP3v(values):
    res = xgbDLP3v.predict(values)
    return res[0]

#2variables: Chol, Tg
def fit_xgbDLP2v(values):
    res = xgbDLP2v.predict(values)
    return res[0]

#[chol,hdl,tg]
###start application###
st.title('Indirect-LDL calculation by XGB-ThaiLDL model')
txt1 = """Calculat indirect-LDL from 2-3 parameters: cholesterol,
triglyceride, HDL"""
st.write(txt1)

###Input values###
st.header('Please insert Cholesterol, HDL, Triglyceride')

cho = st.number_input('Cholesterol', min_value=0, max_value=1000, value=200, step=1)
hdl = st.number_input('HDL-C', min_value=0, max_value=120, value=40, step=1)
tg = st.number_input('Triglyceride', min_value=0, max_value=1000, value=150, step=1)  

#data
info3v = np.array([[cho,hdl,tg]])
info2v = np.array([[cho,tg]])

#predict
ans3v = fit_xgbDLP3v(info3v)
ans2v = fit_xgbDLP2v(info2v)

#show result
if st.button('Calculate', key = 'calculate button'):
    st.header('Indirect-LDL')
    st.subheader('From 3 parameters: Cholesterol, HDL-C, Triglyceride')
    st.write('Indirect-LDL =', round(ans3v,2), 'mg/dL')
    st.subheader('From 2 parameters: Cholesterol, Triglyceride')
    st.write('Indirect-LDL =', round(ans2v,2), 'mg/dL')

### explaination
#sidebar
st.sidebar.header('XGB-ThaiLDL model')
txt2 = """Dr. Padeomwut Teerawongsakul inspires this project. The dataset for training and testing the models to calculate indirect LDL cholesterol was from Vajira Hospital's lipid data which has 54144 profiles. We ([VBaM4H](https://www.vbam4h.com/)) experimented with many machine learning models on this dataset. The results show that the best is the Extreme Gradient Boosting (XGBoost) model. All experiments, model development, and web application are performed by [Dr. Wanjak Pongsittisak](mailto:wanjak@nmu.ac.th). The complete experiment method and results will be prepared and published soon. However, The performance of models is partially reported. For more information or any suggestion, don't hesitate to get in touch with [us](mailto:wanjak@nmu.ac.th)."""
st.sidebar.write(txt2)
st.sidebar.write('Version: Beta')
st.sidebar.markdown('&copy; 2022 VBaM4H All Rights Reserved')

#text below
#table
performance = {'RMSE': [11.12,15.27,26.55,24.11],
              'Pearson r': [0.955,0.913,0.807,0.818]}

table1 = pd.DataFrame(performance, index = ['XGB-ThaiLDL 3 parameters', 'XGB-ThaiLDL 2 parameters','Friedewald equation','Martin-Hopkins equation'])
st.write('---------------------------------------------------------------------')
st.header('Results')
txt3 = """We recommend using the XGB-ThaiLDL 3 parameters model, which is betters than with 2 parameters. However, the XGB-ThaiLDL 2 parameters model is for some limited-resource settings. Our two models' performance is superior to the previous conventional equations on the Thai population. The performance is shown below."""
st.write(txt3)
st.dataframe(table1)

#thank you
#thank you
st.header('Credit')
txt4 = '''All code is written by [Python 3.7] (https://www.python.org/). We thank everyone who contributes to all libraries and packages that we used: [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [Matplotlib](https://matplotlib.org/), and [Streamlit](https://streamlit.io/)'''
st.write(txt4)
