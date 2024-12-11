#!/usr/bin/env python
# coding: utf-8

#  ### Dataset Info: Sample Data Set containing Telco customer data and showing customers left last month

# In[1]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the data file
# 

# In[2]:


telco_base_data = pd.read_csv("C:\\Users\\chetan\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


telco_base_data.head()


# In[4]:


telco_base_data.shape


# In[5]:


telco_base_data.columns.values


# In[6]:


# Checking the data types of all the columns
telco_base_data.dtypes


# In[7]:


# Check the descriptive statistics of numeric variables
telco_base_data.describe()


# SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not propoer
# 
# 75% customers have tenure less than 55 months
# 
# Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# In[8]:


telco_base_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[9]:


100*telco_base_data['Churn'].value_counts()/len(telco_base_data['Churn'])


# In[10]:


telco_base_data['Churn'].value_counts()


# Data is highly imbalanced, ratio = 73:27
# So we analyse the data with other features while taking the target values separately to get some insights.

# In[11]:


# Concise Summary of the dataframe, as we have too many columns, we are using the verbose = True mode
telco_base_data.info(verbose = True) 


# In[12]:


missing = pd.DataFrame((telco_base_data.isnull().sum()) * 100 / telco_base_data.shape[0]).reset_index()
missing.columns = ['Column', 'MissingPercentage']

# Plot
plt.figure(figsize=(16, 5))
sns.pointplot(x='Column', y='MissingPercentage', data=missing)
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing Values")
plt.ylabel("Percentage")
plt.show()


# # Data Cleaning

# 1. Create a copy of base data for manupulation & processing

# In[13]:


telco_data = telco_base_data.copy()


# 2. Total Charges should be numeric amount. Let's convert it to numerical data type

# In[15]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()


# 3. As we can see there are 11 missing values in TotalCharges column. Let's check these records

# In[16]:


telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# 4. Missing Value Treatement

# Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

# In[17]:


#Removing missing values 
telco_data.dropna(how = 'any', inplace = True)


# 5. Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[20]:


# Get the max tenure
print(telco_data['tenure'].max()) 


# In[21]:


# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# In[22]:


telco_data['tenure_group'].value_counts()


# In[23]:


# 6. Remove columns not required for processing

#drop column customerID and tenure
telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()


# # Data Exploration
# 

# # 1.  Plot distibution of individual predictors by churn

# # Univariate Analysis

# In[26]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


# In[27]:


#2. Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0

telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)
telco_data.head()


# In[28]:


# Convert all the categorical variables into dummy variables

telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()


# In[29]:


#Relationship between Monthly Charges and Total Charges

sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# Total Charges increase as Monthly Charges increase - as expected.

# In[31]:


#Churn by Monthly Charges and Total Charges

Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# Insight: Churn is high when Monthly Charges ar high
# 

# In[33]:


Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# **Surprising insight**  as higher Churn at lower Total Charges
# 
# However if we combine the insights of 3 parameters i.e. Tenure, Monthly Charges & Total Charges then the picture is bit clear :- Higher Monthly Charge at lower tenure results into lower Total Charge. Hence, all these 3 factors viz Higher Monthly Charge, Lower tenure and Lower Total Charge are linkd to High Churn.
# 
# 

# In[34]:


#11. Build a corelation of all predictors with 'Churn'

plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# **Derived Insight:**
# 
# HIGH Churn seen in case of Month to month contracts, No online security, No Tech support, First year of subscription and Fibre Optics Internet
# 
# LOW Churn is seens in case of Long term contracts, Subscriptions without internet service and The customers engaged for 5+ years
# 
# Factors like Gender, Availability of PhoneService and # of multiple lines have alomost NO impact on Churn
# 
# This is also evident from the Heatmap below

# In[35]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(), cmap="Paired")


# # Bivariate Analysis

# In[36]:


new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]


# In[37]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[38]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[39]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[40]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[41]:


uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[42]:


uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[43]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# #  ****CONCLUSION****

# These are some of the quick insights from this exercise:
# 
# 1.Electronic check medium are the highest churners
# 2.Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.
# 3.No Online security, No Tech Support category are high churners
# 4.Non senior Citizens are high churners
# 
# 
# Note: There could be many more such insights, so take this as an assignment and try to get more insights :)

# In[ ]:




