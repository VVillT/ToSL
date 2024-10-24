import streamlit as st
import pandas as pd
import sklearn as skl
from sklearn.ensemble import GradientBoostingClassifier
import joblib
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
st.title('Titanic Survival Analysis and Prediction')

st.subheader('Making Prediction')
st.markdown('**Please provide information of the passenger and we will see how likely he will survive**:')  # you can use markdown like this

# load models
tree_clf = joblib.load('rf_optimal.pickle')

# get inputs

sex = st.selectbox('Sex', ['female', 'male'])
age = int(st.number_input('Age:', 0, 120, 20))
sib_sp = int(st.number_input('# of siblings / spouses aboard:', 0, 10, 0))
par_ch = int(st.number_input('# of parents / children aboard:', 0, 10, 0))
pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
embarked = st.selectbox('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

passenger = pd.DataFrame(
    {
        'Pclass': [pclass],
        'Sex': [sex], 
        'Age': [age],
        'SibSp': [sib_sp],
        'Parch': [par_ch],
#        'Fare': [fare],
        'Embarked': [embarked],
    }
)

def transform_passenger_data(passenger_data):
    # Initialize transformed data as a dictionary
    transformed_data = {
        'Age': passenger_data['Age'][0] if passenger_data['Age'] is not None else np.nan,
        'SibSp': passenger_data['SibSp'][0],
        'Parch': passenger_data['Parch'][0],
        'Sex_male': 1 if passenger_data['Sex'][0] == 'male' else 0,
        'Embarked_Q': 1 if passenger_data['Embarked'][0] == 'Q' else 0,
        'Embarked_S': 1 if passenger_data['Embarked'][0] == 'S' else 0,
        'Pclass_2': 1 if passenger_data['Pclass'][0] == 2 else 0,
        'Pclass_3': 1 if passenger_data['Pclass'][0] == 3 else 0
    }
    
    # Convert the dictionary to a DataFrame with the correct column order
    df_transformed = pd.DataFrame([transformed_data])
    
    return df_transformed

transformed_passenger = transform_passenger_data(passenger)
transformed_passenger



y_pred = tree_clf.predict(passenger)

if y_pred[0] == 0:
    msg = 'This passenger is predicted to be: **died**'
else:
    msg = 'This passenger is predicted to have {:.2f} chance to be:be: **survived**'.format(predtest[1]*100)

st.write(msg)
