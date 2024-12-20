# Customer-Churn-Anaylsis

# Aim and Objective
The objective of this project is to create a model that effectively predicts customer churn, allowing businesses to identify at-risk customers. By pinpointing potential churners in advance, companies can deploy specific retention tactics to minimize customer loss, enhance satisfaction, and boost profitability.

# Flow of the Project
Business Understanding --> Data Understanding --> Data Preprocessing --> Modelling --> Evaluation

# Data Understanding

The dataset contained the following features:

- Gender -- Whether the customer is a male or a female
- SeniorCitizen -- Whether a customer is a senior citizen or not
- Partner -- Whether the customer has a partner or not (Yes, No)
- Dependents -- Customer has dependents or not (Yes, No)
- Tenure -- Number of months the customer has stayed with the company
- Phone Service -- Whether the customer has a phone service or not (Yes, No)
- MultipleLines -- Whether the customer has multiple lines or not
- InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)
- OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)
- OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)
- DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)
- TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)
- StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)
- StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)
- Contract -- The contract term of the customer (Month-to-Month, One year, Two year)
- PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)
- Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
- MonthlyCharges -- The amount charged to the customer monthly
- TotalCharges -- The total amount charged to the customer
- Churn -- Whether the customer churned or not (Yes or No)
  
# Data Preprocessing

- Used Simple Imputer using the mean to Impute the missing values
- Used label encoder for the target column(Churn column) to encode the two classes 'Yes' and 'No'
- Used One Hot Encoder to encode the other remaining categorical columns
- Used SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in target variable(Churn column).
- Used Standard scaler to normalize the dataset
 
# Modelling
The following Models were trained:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- XGBoost
- LightGBM

