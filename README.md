🤖 High-Accuracy Salary Prediction App
A complete end-to-end machine learning project that predicts employee salaries using a highly accurate Random Forest model and deploys it as an interactive web application with Streamlit.

Note: To add your own screenshot, take a picture of your running app, upload it to a site like Imgur, and replace the URL above with your image link.

✨ Key Features
High-Accuracy Predictions: The final model achieves an R-squared of 0.92, meaning it can explain 92% of the variation in salary.

Advanced Feature Engineering: Goes beyond a basic model by creating intelligent features (Job_Level and Experience_x_Level) to capture real-world business logic and improve performance.

Interactive Web App: A user-friendly interface built with Streamlit that allows anyone to get salary predictions easily.

Complete ML Workflow: Demonstrates the full end-to-end process from data cleaning and analysis to model deployment.

🛠️ Tech Stack
This project utilizes a modern, industry-standard tech stack for data science and web deployment:

🚀 Getting Started
Follow these steps to get the project running on your local machine.

1. Prerequisites
Make sure you have Python 3.8+ and pip installed on your system.

2. Clone the Repository
git clone https://github.com/your-username/ml-salary-prediction-app.git
cd ml-salary-prediction-app

3. Install Dependencies
This project uses a requirements.txt file to manage its dependencies. Run the following command to install them:

pip install -r requirements.txt

4. Train the Model
Before you can run the web app, you need to train the machine learning model. The Jupyter Notebook automates this process.

Start Jupyter:

jupyter notebook

In the browser window that opens, navigate to notebooks/ and open salary_prediction_model.ipynb.

Click on Cell > Run All from the menu bar.

This will train the final model and save the necessary files (.pkl) into the saved_models/ directory.

5. Launch the Web App!
Once the model is trained, you can launch the Streamlit application.

streamlit run app.py

Your web browser will open a new tab with the live application!

📈 Model Performance
The final model selected was a Random Forest Regressor, which significantly outperformed the baseline models after feature engineering.

Metric

Final Model Score

R-squared (R²)

0.92

Mean Absolute Error

$8,542.65

This high R-squared value indicates that the model's features are excellent at explaining the differences in salary.

🌟 Future Improvements
This project provides a solid foundation. Future enhancements could include:

Hyperparameter Tuning: Use techniques like GridSearchCV to find the optimal parameters for the Random Forest model.

Add More Features: Incorporate Location as a feature, as this is a major factor in salary differences.

Expand the Dataset: Train the model on a larger, more recent dataset to improve its real-world accuracy.