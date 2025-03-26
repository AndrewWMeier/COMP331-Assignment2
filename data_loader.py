# data_loader.py - Functions to load and preprocess the Adult Income dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_adult_dataset():
    """
    Load the Adult Income dataset from UCI repository
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", 
        "marital-status", "occupation", "relationship", "race", 
        "gender", "capital-gain", "capital-loss", "hours-per-week", 
        "native-country", "income"
    ]
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, header=None, names=column_names, sep=', ', engine='python')
    
    #clean the data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    #convert income to binary target
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for machine learning:
    - Handle categorical features
    - Scale numerical features
    - Prepare features and target
    """
    # copy to not mess original
    data = df.copy()
    
    # define numerical and categorical columns
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'gender', 'native-country']
    
    #extract target
    y = data['income'].values
    
    #preprocessing for numerical data
    numerical_data = data[numerical_cols].copy()
    
    #standardize numerical features
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(numerical_data)
    
    #preprocessing for categorical data
    categorical_data = data[categorical_cols].copy()
    
    #one-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(categorical_data)
    
    #get feature names after one-hot encoding
    feature_names = numerical_cols.copy()
    for i, col in enumerate(categorical_cols):
        encoded_names = [f"{col}_{val}" for val in encoder.categories_[i]]
        feature_names.extend(encoded_names)
    
    #combine processed features
    X = np.hstack([numerical_data, categorical_data])
    
    return X, y, feature_names