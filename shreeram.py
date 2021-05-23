import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout='wide')
st.sidebar.write("""
This app predicts the **Profit** 
on 

[50 Startups Dataset](https://www.kaggle.com/farhanmd29/50-startups?select=50_Startups.csv)

by using Multiple Linear Regression
***
""")

st.sidebar.header('Provide Input Features here:')


def user_input_features():
    rd = st.sidebar.slider('R&D (in USD)', 100,6000,2000)
    admin = st.sidebar.slider('Administration (in USD)', 100,6000,1500)
    marketing = st.sidebar.slider('Marketing (in USD)', 100,6000,1800)
    state = st.sidebar.selectbox('State',('New York','California','Florida'))
    data = {'R&D Spend': rd,
            'Administration': admin,
            'Marketing Spend': marketing,
            'State': state
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df_raw = pd.read_csv('50_Startups.csv')
df1 = df_raw.drop(columns=['Profit'])
df = pd.concat([input_df,df1],axis=0)

# Encoding of ordinal features
df=pd.get_dummies(df,columns=['State'],drop_first=True)
df=df[:1]

st.write("""
         # Aditya app
         ## Startup Profit Prediction 
""")
# Displays the user input features
st.subheader('User Input features')

st.write(df)

# Reads in saved classification model
model = pickle.load(open('lr.pkl', 'rb'))

# Apply model to make predictions
prediction = model.predict(df)
#prediction_proba = load_clf.predict_proba(df)

st.markdown("""
            ***
### Predicted Profit
for selected features (in USD) will be around: 
""")

st.subheader('')
st.write(prediction)

st.markdown("""
***
* for this code and much more, Follow me on [Aditya Github](https://github.com/aditya11ad)
""")
