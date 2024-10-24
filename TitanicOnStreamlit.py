import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Random Forest Classifier, SimpleImputer, train_test_split, GridSearchCV and metrics from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
st.title('Titanic Survival Analysis and Prediction')

st.subheader('Making Prediction')
st.markdown('**Please provide information of the passenger and we will see how likely he will survive**:')  # you can use markdown like this



#---------------------------------#
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
testpt2 = pd.read_csv("gender_submission.csv")
testfull = pd.merge(test, testpt2 , on= "PassengerId")

#---------------------------------#
y_train = pd.DataFrame(train['Survived'])
x_train = train[['Age', 'SibSp', 'Parch' , 'Sex', 'Embarked' ,'Pclass']]

# Rebuilding pickle without pipelines:
num_features = ['Age', 'SibSp', 'Parch']
cat_features = ['Sex','Embarked','Pclass']

#Create transformation code so that Streamlit wont be dependent on Pipeline which is causing issues. 
for n in num_features:
	x_train[n] = x_train[n].fillna(x_train[n].mean())
for c in cat_features:
	x_train = pd.get_dummies(x_train, columns=[c], drop_first=True)
st.write(x_train.columns)      

# Random Forest Classifier
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)
st.cache()

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


# get inputs
cols = st.columns(2)

with cols[0]:
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

	transformed_passenger = transform_passenger_data(passenger)

	y_pred = rf.predict(transformed_passenger)
	predtest = rf.predict_proba(transformed_passenger)

with cols[1]:
	if y_pred[0] == 0:
	    msg = f"This passenger is predicted to have {predtest[0][0]*100:.2f}% chance to be: **died**"
	else:
	    msg = f"This passenger is predicted to have {predtest[0][1]*100:.2f}% chance to be: **survived**"
	
	st.write(msg)
