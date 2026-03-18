import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
   
 
st.set_page_config(page_title="Gold Price Prediction",layout="centered")

st.title("Gold Price Prediction using Machine Learning")
st.write("Predict gold prices based on economic indicators using Random Forest model.")
  
@st.cache_data
def load_data():
    path=r"C:\Users\artis\Downloads\gld_price_data.csv"
    return pd.read_csv(path)

df=load_data()

if st.checkbox("Show Dataset"):
    st.write(df.head())

st.subheader("Feature Correlation")
numeric_df= df.select_dtypes(include=[np.number])

fig, ax =plt.subplots(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".1f",
    cmap="Blues",
    square=True,
    cbar=True,
    ax=ax
)
st.pyplot(fig)

st.subheader("Gold Price Distribution")
fig2,ax2=plt.subplots()
sns.histplot(df['GLD'],kde=True,ax=ax2)
st.pyplot(fig2)

X=df.drop(['Date','GLD'],axis=1)
Y=df['GLD']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=2)

model=RandomForestRegressor(n_estimators=100,random_state=2)
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)
score=r2_score(Y_test,y_pred)

st.subheader("Model Performance")
st.write(f"R2 Score :{score:.3f}")

fig3,ax3=plt.subplots()
ax3.plot(list(Y_test),label="Actual Price")
ax3.plot(y_pred,label="Predicted Price")
ax3.set_title("Actual vs Predcited Gold Price")
ax3.legend()
st.pyplot(fig3)

st.subheader("Predict Gold Price")

SPX=st.number_input("S&P 500 Index (SPX)",value=1800.0)
USO=st.number_input("Oil Price (USO)",value=25.0)
SLV=st.number_input("Silver Price (SLV)",value=1.2)
EUR_USD=st.number_input("EUR/USD Exchange Rate ",value=0.5)

if st.button("Predict Price"):
    input_data=np.asarray([SPX,USO,SLV,EUR_USD]).reshape(1,-1)
    prediction=model.predict(input_data)
    st.success(f"Predicted Gold Price :{prediction[0]:.2f}")