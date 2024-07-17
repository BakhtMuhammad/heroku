import streamlit as st
import pickle
import numpy as np

st.title("Titanic Survival Prediction App")

# Load the model
with open('titanicclassifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Define the prediction function
def prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Convert inputs to appropriate types
    Pclass = int(Pclass)
    Sex = 1 if Sex == 'male' else 0
    Age = float(Age)
    SibSp = int(SibSp)
    Parch = int(Parch)
    Fare = float(Fare)
    Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]

    # Predict
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = classifier.predict(features)
    return prediction

def main():
    st.title("Titanic Survival Prediction App")

    # Input fields
    Pclass = st.text_input("Passenger Class (1, 2, or 3)")
    Sex = st.selectbox("Sex", ('male', 'female'))
    Age = st.text_input("Age")
    SibSp = st.text_input("Number of Siblings/Spouses Aboard")
    Parch = st.text_input("Number of Parents/Children Aboard")
    Fare = st.text_input("Fare")
    Embarked = st.selectbox("Port of Embarkation", ('C', 'Q', 'S'))

    if st.button("Predict"):
        try:
            result = prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
            st.success(f'This output is: {"Survived" if result[0] == 1 else "Did not survive"}')
        except Exception as e:
            st.error(f'Error: {e}')

if __name__ == "__main__":
    main()
