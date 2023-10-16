
import gradio as gr
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gradio

# Load model
with open('src/churn_model.pkl', 'rb') as file:
    rfc_loaded = pickle.load(file)

# Load preprocessor
with open('src/churn_pipeline.pkl', 'rb') as file:
    preprocessor_loaded = pickle.load(file)

# Creating the preprocessing pipeline
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']

numerical_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']

categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

def predict(SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure):

    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
             MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    processed_data = preprocessor_loaded.transform(new_df)
    single = rfc_loaded.predict(processed_data)

    if single == 1:
        churn_result = "This customer is likely to be churned."
    else:
        churn_result = "This customer is likely to continue."
    
    return churn_result

# Initialize Gradio input components
iface = gr.Interface(fn=predict,
                     inputs=[
                         gr.Radio(label='Are you a Seniorcitizen; No=0 and Yes=1', choices=[0, 1]),
                         gr.Number(label='Enter Monthly Charges'),
                         gr.Number(label='Enter Total Charges'),
                         gr.Radio(label='Select your Gender', choices=["Male", "Female"]),
                         gr.Radio(label='Do you have a Partner', choices=["Yes", "No"]),
                         
                         gr.Radio(label='Do you have any Dependents? ', choices=["Yes", "No"]),
                         gr.Radio(label='Do you have PhoneService?', choices=["Yes", "No"]),
                         gr.Dropdown(label='Do you have MultipleLines', choices=["Yes", "No", "No phone service"]),
                         gr.Dropdown(label='Do you have InternetService', choices=["DSL", "Fiber optic", "No"]),
                         gr.Dropdown(label='Do you have OnlineSecurity', choices=["Yes", "No", "No internet service"]),
                         gr.Dropdown(label='Do you have OnlineBackup', choices=["Yes", "No", "No internet service"]),
                         
                         gr.Dropdown(label='Do you have DeviceProtection', choices=["Yes", "No", "No internet service"]),
                         gr.Dropdown(label='Do you have TechSupport', choices=["Yes", "No", "No internet service"]),
                         gr.Dropdown(label='Do you have StreamingTV', choices=["Yes", "No", "No internet service"]),
                         gr.Dropdown(label='Do you have StreamingMovies', choices=["Yes", "No", "No internet service"]),
                         gr.Dropdown(label='What is your Contract with Telco', choices=["Month-to-month", "One year", "Two year"]),
                         gr.Radio(label='Do you prefer PaperlessBilling?', choices=["Yes", "No"]),
                         gr.Dropdown(label='Which PaymentMethod do you prefer?', choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
                         gr.Slider(minimum=0, maximum=72)
                     ],


                     outputs=gr.Label(label='Churn'),
                     theme="monochrome",
                     max_width=800)

iface.launch(share=True, debug=True)
