import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title('Titanic Survival Analysis and Prediction')

st.subheader('Making Prediction')
st.markdown('**Please provide information of the passenger and we will see how likely he will survive**:')  # you can use markdown like this

# load models
tree_clf = joblib.load('clf-best.pickle')

# get inputs

sex = st.selectbox('Sex', ['female', 'male'])
age = int(st.number_input('Age:', 0, 120, 20))
sib_sp = int(st.number_input('# of siblings / spouses aboard:', 0, 10, 0))
#par_ch = int(st.number_input('# of parents / children aboard:', 0, 10, 0))
pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
fare = int(st.number_input('# of parents / children aboard:', 0, 100, 0))
#embarked = st.selectbox('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

# this is how to dynamically change text
prediction_state = st.markdown('calculating...')

passenger = pd.DataFrame(
    {
        'Pclass': [pclass],
        'Sex': [sex], 
        'Age': [age],
        'SibSp': [sib_sp],
#        'Parch': [par_ch],
        'Fare': [fare],
#        'Embarked': [embarked],
    }
)

y_pred = tree_clf.predict(passenger)

if y_pred[0] == 0:
    msg = 'This passenger is predicted to be: **died**'
else:
    msg = 'This passenger is predicted to be: **survived**'

prediction_state.markdown(msg)