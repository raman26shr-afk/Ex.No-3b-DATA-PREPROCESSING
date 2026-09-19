# Ex.No-3b-DATA PREPROCESSING
## Aim
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset into training and testing sets, and applying feature scaling.

# THEORY

Data preprocessing is an essential step in Artificial Intelligence and Machine Learning. It involves transforming raw data into a clean, consistent, and suitable format for building machine learning models.

Real-world datasets may contain missing values, categorical information, and numerical features with different scales. Preprocessing helps improve data quality and makes the dataset suitable for machine learning algorithms.

# OBJECTIVE

To preprocess a dataset using Python and Scikit-learn by handling missing values, encoding categorical variables, splitting the dataset into training and testing sets, and applying feature scaling to prepare the data for machine learning applications.

# Importance of Data Preprocessing

Handling Missing Values: Missing data can affect the accuracy of machine learning models. Imputation techniques can be used to replace missing values.

Encoding Categorical Data: Machine learning algorithms generally require numerical inputs. Categorical values such as country names must be converted into numerical representations.

Splitting the Dataset: Dividing data into training and testing sets helps evaluate model performance on unseen data.

Feature Scaling: Scaling ensures that numerical features with different ranges contribute appropriately to algorithms that are sensitive to feature magnitude.

Improving Data Quality: Preprocessing reduces inconsistencies and prepares data for further analysis.

# REQUIREMENTS
Hardware Requirements

Computer or laptop.

Minimum 4 GB RAM.

Keyboard and mouse.

Software Requirements

Python 3.x.

Google Colab or Jupyter Notebook.

Pandas.

NumPy.

Scikit-learn.

Google Drive (for storing the dataset).

# PROCEDURE

Import the required Python libraries.

Mount Google Drive and load the dataset using Pandas.

Display the first few records of the dataset.

Inspect the dataset using df.info() and df.shape.

Separate the independent variables (X) and dependent variable (Y).

Convert the independent variables into an array.

Identify and handle missing values using SimpleImputer with the mean strategy.

Encode the categorical Country column using LabelEncoder.

Apply One-Hot Encoding to convert categorical country values into dummy variables.

Encode the dependent variable Purchased using LabelEncoder.

Split the dataset into training and testing sets using train_test_split.

Apply StandardScaler for feature scaling.

Display the preprocessed training and testing datasets.

# PROGRAM

Step 1: Import Libraries and Load Dataset
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/Data.csv'
)

print(df.head())

Step 2: Check Dataset Information
df.info()
print("Dataset Shape:", df.shape)

Step 3: Separate Independent and Dependent Variables
X = df[['Country', 'Age', 'Salary']].values
y = df['Purchased'].values

print("Independent Variables:")
print(X)

print("Dependent Variable:")
print(y)

Step 4: Handle Missing Values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)

X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("After Handling Missing Values:")
print(X)

Step 5: Encode Categorical Data
from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()

X[:, 0] = label_encoder_x.fit_transform(X[:, 0])

print("After Label Encoding:")
print(X)

Step 6: Apply One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

X_country = onehotencoder.fit_transform(
    df[['Country']]
)

print("One-Hot Encoded Country:")
print(X_country)
Encode the Dependent Variable
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

print("Encoded Dependent Variable:")
print(y)

Step 7: Split Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

print("Training Data:")
print(X_train)

print("Testing Data:")
print(X_test)

print("Training Labels:")
print(y_train)

print("Testing Labels:")
print(y_test)

Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print("Scaled Training Data:")
print(X_train)

print("Scaled Testing Data:")
print(X_test)

# APPLICATIONS

Data preparation for classification and regression models.

Customer purchase prediction.

Healthcare data analysis.

Financial risk prediction.

Student performance prediction.

Customer segmentation and recommendation systems.

# RESULT

The given dataset was successfully preprocessed using Python and Scikit-learn. Missing values were handled using mean imputation, categorical variables were encoded, and the dataset was divided into training and testing sets. Feature scaling was also performed to prepare the data for further Machine Learning applications.
