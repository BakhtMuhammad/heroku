import streamlit as st
import pickle

st.title("Welcome to Titanic Prediction App :ship:")
st.image('titanic2.jpg')
pickle_in = open("titanicdataset.pkl", "rb")
classifier = pickle.load(pickle_in)

#Defining the function which will make the predictin using the data that user will input
def prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    prediction = classifier.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    print(prediction)
    return prediction

def main():
    st.title("Titanic Prediction App")

    #The following code creates text boxes in which the user can enter the data required to make prediction
    Pclass = st.text_input("Passenger Class")
    Sex = st.text_input("Sex")
    Age = st.text_input("Age")
    SibSp = st.text_input("Sibling/Spouse")
    Parch = st.text_input("Parent/Child")
    Fare = st.text_input("Fare")
    Embarked = st.text_input("Embarked")
    result = ""

    #This code ensure that whent he button 'Predict' is clicked, the prediction function defined above
    #is classed to make the prediction and store it in the variable result
    if st.button("Predict"):

        #Convert inputs to appropriate types
        Pclass = int(Pclass)
        Age = float(Age)
        SibSp = int(SibSp)
        Parch = int(Parch)
        Fare = float(Fare)

        result = prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    st.success('This output is: {}'.format(result))

main()
