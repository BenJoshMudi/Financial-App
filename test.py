import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_path = "C:/Users/mudia/OneDrive/Desktop/Finclusion/model_saved"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Mapping dictionaries for encoded values
location_type_map = {0: 'Rural', 1: 'Urban'}
cellphone_access_map = {1: 'Yes', 0: 'No'}
gender_map = {0: 'Male', 1: 'Female'}
relationship_with_head_map = {0: 'Head of Household', 1: 'Spouse', 2: 'Child', 3: 'Parent', 4: 'Other relative', 5: 'Other non-relatives'}
marital_status_map = {0: 'Married/Living together', 1: 'Divorced/Seperated', 2: 'Widowed', 3: 'Single/Never Married', 4: 'Don’t know'}
education_level_map = {0: 'No formal education', 1: 'Primary education', 2: 'Secondary education', 3: 'Vocational/Specialised training', 4: 'Tertiary education', 5: 'Other/Dont know/RTA'}
job_type_map = {0: 'Farming and Fishing', 1: 'Self employed', 2: 'Formally employed Government', 3: 'Formally employed Private', 4: 'Informally employed', 5: 'Remittance Dependent', 6: 'Government Dependent', 7: 'Other Income', 8: 'No Income', 9: 'Dont Know/Refuse to answer'}

# Function to convert categorical inputs to numerical
def categorical_to_numerical(value, category_map):
    return category_map.get(value, 0.0)  # Default to 0.0 if value not found

# Create a function for prediction
def expresso_prediction(input_data):
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
    st.title("Bank Account Access")
    
    # Getting the input data from the user using radio buttons
    location_type = st.radio('Type of location', list(location_type_map.values()))
    cellphone_access = st.radio('Cellphone access', list(cellphone_access_map.values()))
    household_size = st.text_input('Number of people living in one house')
    age_of_respondent = st.number_input('Age of the interviewee')
    gender_of_respondent = st.radio('Gender of interviewee:', list(gender_map.values()))
    relationship_with_head = st.radio('The interviewee’s relationship with the head of the house:', list(relationship_with_head_map.values()))
    marital_status = st.radio('The marital status of the interviewee:', list(marital_status_map.values()))
    education_level = st.radio('Highest level of education:', list(education_level_map.values()))
    job_type = st.radio('Type of job interviewee has:', list(job_type_map.values()))
    
    diagnosis = ''
    
    if st.button('Check Eligibility'):
        diagnosis = expresso_prediction([
            location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, 
            relationship_with_head, marital_status, education_level, job_type
        ])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
