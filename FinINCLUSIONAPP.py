import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_path = "C:/Users/mudia/OneDrive/Desktop/Finclusion/model_saved"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Function to convert categorical inputs to numerical
def categorical_to_numerical(value, category_map):
    return category_map.get(value, 0.0)  # Default to 0.0 if value not found

# Create a function for prediction
def expresso_prediction(input_data):
    # Define mappings for categorical variables
    location_type_map = {'Rural': 0.0, 'Urban': 1.0}
    cellphone_access_map = {'Yes': 1.0, 'No': 0.0}
    gender_map = {'Male': 0.0, 'Female': 1.0}
    relationship_with_head_map = {
        'Head of Household': 0.0, 'Spouse': 1.0, 'Child': 2.0, 
        'Parent': 3.0, 'Other relative': 4.0, 'Other non-relatives': 5.0, 
        'Dont know': 6.0
    }
    marital_status_map = {
        'Married/Living together': 0.0, 'Divorced/Seperated': 1.0, 
        'Widowed': 2.0, 'Single/Never Married': 3.0, 'Dont know': 4.0
    }
    education_level_map = {
        'No formal education': 0.0, 'Primary education': 1.0, 
        'Secondary education': 2.0, 'Vocational/Specialised training': 3.0, 
        'Tertiary education': 4.0, 'Other/Dont know/RTA': 5.0
    }
    job_type_map = {
        'Farming and Fishing': 0.0, 'Self employed': 1.0, 
        'Formally employed Government': 2.0, 'Formally employed Private': 3.0, 
        'Informally employed': 4.0, 'Remittance Dependent': 5.0, 
        'Government Dependent': 6.0, 'Other Income': 7.0, 
        'No Income': 8.0, 'Dont Know/Refuse to answer': 9.0
    }
    
    # Convert categorical inputs to numerical
    location_type_num = categorical_to_numerical(input_data[0], location_type_map)
    cellphone_access_num = categorical_to_numerical(input_data[1], cellphone_access_map)
    gender_num = categorical_to_numerical(input_data[4], gender_map)
    relationship_with_head_num = categorical_to_numerical(input_data[5], relationship_with_head_map)
    marital_status_num = categorical_to_numerical(input_data[6], marital_status_map)
    education_level_num = categorical_to_numerical(input_data[7], education_level_map)
    job_type_num = categorical_to_numerical(input_data[8], job_type_map)
    
    # Convert other inputs to float
    household_size = float(input_data[2])
    age_of_respondent = float(input_data[3])
    
    # Create input array for prediction
    input_data_as_num = np.asarray([
        location_type_num, cellphone_access_num, household_size, age_of_respondent, 
        gender_num, relationship_with_head_num, marital_status_num, 
        education_level_num, job_type_num
    ], dtype=float).reshape(1, -1)
    
    # Debugging information
    st.write("Input data as numerical array:", input_data_as_num)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_as_num)
    
    st.write("Model prediction:", prediction)
    
    if prediction[0] == 1:
        return 'The customer has access.'
    else:
        return 'The customer does not have access.'

def main():
    st.title("Bank Account Access Predictor")
    
    # Getting the input data from the user using radio buttons
    location_type = st.radio('Type of location', ['Rural', 'Urban'])
    cellphone_access = st.radio('Cellphone access', ['Yes', 'No'])
    household_size = st.text_input('Number of people living in one house')
    age_of_respondent = st.text_input('Age of the interviewee')
    gender_of_respondent = st.radio('Gender of interviewee:', ['Male', 'Female'])
    relationship_with_head = st.radio('The interviewee’s relationship with the head of the house:', [
        'Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives', 'Dont know'
    ])
    marital_status = st.radio('The marital status of the interviewee:', [
        'Married/Living together', 'Divorced/Seperated', 'Widowed', 'Single/Never Married', 'Don’t know'
    ])
    education_level = st.radio('Highest level of education:', [
        'No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 
        'Tertiary education', 'Other/Dont know/RTA'
    ])
    job_type = st.radio('Type of job interviewee has:', [
        'Farming and Fishing', 'Self employed', 'Formally employed Government', 'Formally employed Private', 
        'Informally employed', 'Remittance Dependent', 'Government Dependent', 'Other Income', 'No Income', 
        'Dont Know/Refuse to answer'
    ])
    
    diagnosis = ''
    
    if st.button('Check Eligibility'):
        diagnosis = expresso_prediction([
            location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, 
            relationship_with_head, marital_status, education_level, job_type
        ])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()


