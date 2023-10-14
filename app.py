# Import necessary libraries
import gradio as gr
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Initialize transformers
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

# Create a preprocessing pipeline
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
        return "This customer is likely to be churned."
    else:
        return "This customer is likely to continue."

input_components = []
with gr.Blocks(theme= "monochrome") as iface:
    
    title="Customer Churn Prediction" 
    
    img = gr.Image("churn.jpg",height='700', width='500')
     
    with gr.Row():
        title = gr.Label("Customer Churn Prediction")
        
    with gr.Row():
        img
        
    with gr.Row():
        gr.Markdown(" # Predict whether a customer is likely to churn.")

    with gr.Row():
        with gr.Column(scale=3, min_width=600):
        
            input_components=[
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
                gr.Slider(minimum=0, maximum=72, label='What is the length of Tenure with Telco : (0-72 months)')

                ]
    
    with gr.Row():
        pred = gr.Button('Predict')
        
    
    
    output_components = gr.Label(label='Churn') 
    
    pred.click(fn=predict,
            inputs=input_components,
            outputs=output_components,
            )        
           
iface.launch(share=True, debug=True)

