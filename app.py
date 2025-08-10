import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# load model and scaler
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')


# title
st.title("Customer Churn Predictor")
st.write(
    """
    Enter customer account details below to predict the likelihood of churn.  
    The prediction is based on a machine learning model trained on historical customer data.
    """
)

st.caption("📊 Predict whether a customer will stay or leave based on their usage patterns and account details.")

# input fields:
# Account Weeks

# account_weeks = st.number_input('How many weeks has this person been a customer for?', min_value=0, value=0)

# ContractRenewal (Yes / No -> 1 | 0)
col1, col2 = st.columns(2)
with col1:
    contract_renewal = st.radio('Has renewed contract?', options=['Yes', 'No'])
    contract_renewal_binary = 1 if contract_renewal == 'Yes' else 0


# DataPlan (Yes / No -> 1 | 0)    
with col2:
    data_plan = st.radio('Purchased a data plan?', options=['Yes', 'No'])
    data_plan_binary = 1 if data_plan=='Yes' else 0

# CustomerServiceCalls
cust_serv_calls = st.number_input('Number of calls to customer service in the last 30 days:', min_value=0, value=0)

# DayMins
col3, col4 = st.columns(2)
with col3:
    day_mins = st.number_input("Minutes customer spends on calls per day:", min_value=0, value=0)

with col4:
# RoamMins(put it above for better UI appearance)
    roam_mins = st.number_input("Roaming minutes per month:", min_value=0, value=0)

# avg number of daytime calls
# with col4:
#     day_calls = st.number_input("Average number of daytime calls:", min_value=0, value=0)

# monthly charge
monthly_charge = st.number_input("Monthly bill of customer (in USD):", min_value=0.0, value=0.0)

# overage fee
col5, col6 = st.columns(2)

# with col5:
#     has_overage = st.selectbox("Does customer have any overage fee?", options=["Yes", "No"], index=1)
# with col6:
#     if has_overage=="Yes":
#         overage_fee = st.number_input("Largest overage fee in the Last 12 months: ", min_value=0.0, value=0.0)
#     else:
#         overage_fee = 0.0


usage_value = monthly_charge*day_mins
serivice_stress = cust_serv_calls*roam_mins

# creating a DataFrame with User inputs:
user_data = pd.DataFrame({
    'ContractRenewal': [contract_renewal_binary],
    'DataPlan': [data_plan_binary],
    'CustServCalls': [cust_serv_calls],
    'DayMins': [day_mins],
    'MonthlyCharge': [monthly_charge],
    'RoamMins': [roam_mins],
        'ServiceStress': [serivice_stress],
    'UsageValue': [usage_value],
})



# Make predictions
# prediction = model.predict(user_data)
# probability = model.predict_proba(user_data)[:, 1]

if st.button('Predict'):
    with st.spinner("Loading..."):

        #make prediction
        prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)[:, 1]

        st.subheader("Model Predictions")

        st.write("Likely to Churn: ", "Yes" if prediction[0]==1 else "No")
            #st.write("Is this customer likely to Churn: ", "Yes" if prediction[0]==1 else "No")

        #with col23:    
            #st.write(f"Probability of Churning: {probability[0]*100: .2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[0] * 100,  # Convert probability to percentage
            title={'text': "Probability of churning"},
            number={'suffix':"%"},
            gauge={
                'axis': {'range': [0, 100]},  # Change range to 0-100 for percentage
                'bar': {'color': "black"},
                'bgcolor': "lightgray",
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': probability[0] * 100
                }
            }
        ))

        # Set size of the gauge plot
        fig.update_layout(
            height=300,  # Set height here
            width=300    # Set width here
        )

        st.plotly_chart(fig)


