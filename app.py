import streamlit as st
import pandas as pd
import pickle

# --- 1. Set Page Configuration (must be the first Streamlit command) ---
st.set_page_config(page_title="Salary Predictor", page_icon="💵")


# --- 2. Load Model and Columns ---
# Use a function with caching to load the model only once
@st.cache_resource
def load_model_and_columns():
    try:
        with open('saved_models/salary_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('saved_models/model_columns.pkl', 'rb') as columns_file:
            model_columns = pickle.load(columns_file)
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_model_and_columns()


# --- 3. Feature Engineering Function (must match the notebook) ---
# This function assigns a numerical level based on keywords in the job title.
def assign_job_level(job_title):
    job_title = str(job_title).lower()
    if 'director' in job_title: return 5
    elif 'senior' in job_title or 'manager' in job_title: return 4
    elif 'analyst' in job_title: return 3
    elif 'engineer' in job_title or 'developer' in job_title: return 2
    else: return 1


# --- 4. App Layout and Title ---
st.title("💵 Salary Prediction App")
st.write(
    "Enter the details of an employee to predict their potential salary. "
    "This app uses a highly accurate Random Forest model with advanced feature engineering."
)
st.write("---")


# --- 5. Main App Logic ---
# Check if the model is loaded. If not, show an error.
if model is None or model_columns is None:
    st.error(
        "**Model not found!** Please run the Jupyter Notebook "
        "(`notebooks/salary_prediction_analysis.ipynb`) first to train the model "
        "and generate the necessary files."
    )
else:
    # --- User Input Fields ---
    st.header("Enter Employee Details")
    
    # List of common job titles for the dropdown
    job_title_options = [
        'Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 
        'Director', 'Marketing Analyst', 'Product Manager', 'Financial Analyst', 
        'Human Resources Manager', 'Operations Manager', 'Accountant', 'Data Scientist',
        'Marketing Manager', 'Junior Developer', 'Senior Consultant'
    ]

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=70, value=30, step=1)
        education = st.selectbox('Education Level', ["Bachelor's", "Master's", "PhD"])
        # Use a selectbox instead of text_input for Job Title
        job_title = st.selectbox('Job Title', options=job_title_options)

    with col2:
        experience = st.number_input('Years of Experience', min_value=0, max_value=40, value=5, step=1)
        gender = st.selectbox('Gender', ['Male', 'Female'])

    # --- Prediction Button and Logic ---
    if st.button('Predict Salary', type="primary"):
        
        # Create a DataFrame from the user's input
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Education Level': education,
            'Job Title': job_title,
            'Experience': experience
        }])

        # --- Apply the SAME Feature Engineering as the Notebook ---
        
        # 1. Create Job_Level
        input_df['Job_Level'] = input_df['Job Title'].apply(assign_job_level)
        
        # 2. Create Experience_x_Level
        input_df['Experience_x_Level'] = input_df['Experience'] * input_df['Job_Level']
        
        # 3. Drop the original Job Title column
        input_df_engineered = input_df.drop('Job Title', axis=1)
        
        # 4. One-Hot Encode the remaining categorical columns
        input_df_encoded = pd.get_dummies(input_df_engineered, columns=['Gender', 'Education Level'], drop_first=True)
        
        # 5. Align columns with the model's training columns
        input_df_final = input_df_encoded.reindex(columns=model_columns, fill_value=0)

        # --- Make the Prediction ---
        prediction = model.predict(input_df_final)

        # --- Display the Result ---
        st.write("---")
        st.header("Predicted Salary")
        st.success(f'**${prediction[0]:,.2f} per year**')
