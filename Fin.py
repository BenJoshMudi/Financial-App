import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("C:/Users/mudia/OneDrive/Desktop/Finclusion/model_saved", 'rb'))

# Function to convert categorical inputs to numerical
def categorical_to_numerical(value, category_map):
    return category_map.get(value, 0.0)  # Default to 0.0 if value not found

# Create a function for prediction
def expresso_prediction(input_data):
    # Define mappings for categorical variables
    location_type_map = {"Rural": 1, "Urban": 2}
    cellphone_access_map ={"Yes": 1, "No": 0}
    gender_map = {"Female": 1, "Male": 2}

    relationship_with_head_map = {
    "Spouse": 1,
    "Head of Household": 2,
    "Other relative": 3,
    "Child": 4,
    "Parent": 5,
    "Other non-relatives": 6
}
    marital_status_map = {
    "Married/Living together": 1,
    "Widowed": 2,
    "Single/Never Married": 3,
    "Divorced/Seperated": 4,
    "Dont know": 5
}
    education_level_map = {
    "Secondary education": 1,
    "No formal education": 2,
    "Vocational/Specialised training": 3,
    "Primary education": 4,
    "Tertiary education": 5,
    "Other/Dont know/RTA": 6
}
    job_type_map =  {
    "Self employed": 1,
    "Government Dependent": 2,
    "Formally employed Private": 3,
    "Informally employed": 4,
    "Formally employed Government": 5,
    "Farming and Fishing": 6,
    "Remittance Dependent": 7,
    "Other Income": 8,
    "Dont Know/Refuse to answer": 9,
    "No Income": 10
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
    
    # Make prediction
    prediction = loaded_model.predict(input_data_as_num)
    
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
    gender_of_respondent = st.radio('Gender of interviewee:', ['Female', 'male'])
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
