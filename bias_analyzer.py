
import numpy as np
import pandas as pd


def analyze_gender_bias(X, y, gender_idx):
    """
    Analyze gender bias in the dataset
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    gender_idx : int
        Index of the gender_Female feature
        
    Returns:
    --------
    dict
        Report of gender bias statistics
    """
    #extract gender feature (1 for female, 0 for male)
    gender = X[:, gender_idx]
    
    #compute overall statistics
    female_count = np.sum(gender)
    male_count = len(gender) - female_count
    
    female_rate = female_count / len(gender)
    male_rate = male_count / len(gender)
    
    #compute income statistics by gender
    female_high_income = np.sum(y[gender == 1])
    male_high_income = np.sum(y[gender == 0])
    
    female_high_income_rate = female_high_income / female_count if female_count > 0 else 0
    male_high_income_rate = male_high_income / male_count if male_count > 0 else 0
    
    income_rate_ratio = female_high_income_rate / male_high_income_rate if male_high_income_rate > 0 else float('inf')
    
    #create bias report
    bias_report = {
        'female_count': int(female_count),
        'male_count': int(male_count),
        'female_rate': float(female_rate),
        'male_rate': float(male_rate),
        'female_high_income_count': int(female_high_income),
        'male_high_income_count': int(male_high_income),
        'female_high_income_rate': float(female_high_income_rate),
        'male_high_income_rate': float(male_high_income_rate),
        'income_rate_ratio': float(income_rate_ratio),
        'bias_delta': float(male_high_income_rate - female_high_income_rate)
    }
    
    return bias_report

def compute_bias_metrics(y_true, y_pred, gender):
    """
    Compute fairness metrics for model predictions
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    gender : array-like
        Gender values (1 for female, 0 for male)
        
    Returns:
    --------
    dict
        Fairness metrics
    """
    #separate predictions by gender
    female_mask = gender == 1
    male_mask = gender == 0
    
    #calculate true positives, false negatives etc
    female_tp = np.sum((y_true[female_mask] == 1) & (y_pred[female_mask] == 1))
    female_fp = np.sum((y_true[female_mask] == 0) & (y_pred[female_mask] == 1))
    female_tn = np.sum((y_true[female_mask] == 0) & (y_pred[female_mask] == 0))
    female_fn = np.sum((y_true[female_mask] == 1) & (y_pred[female_mask] == 0))
    
    male_tp = np.sum((y_true[male_mask] == 1) & (y_pred[male_mask] == 1))
    male_fp = np.sum((y_true[male_mask] == 0) & (y_pred[male_mask] == 1))
    male_tn = np.sum((y_true[male_mask] == 0) & (y_pred[male_mask] == 0))
    male_fn = np.sum((y_true[male_mask] == 1) & (y_pred[male_mask] == 0))
    
    #calculate fairness metrics
    female_tpr = female_tp / (female_tp + female_fn) if (female_tp + female_fn) > 0 else 0
    male_tpr = male_tp / (male_tp + male_fn) if (male_tp + male_fn) > 0 else 0
    
    female_fpr = female_fp / (female_fp + female_tn) if (female_fp + female_tn) > 0 else 0
    male_fpr = male_fp / (male_fp + male_tn) if (male_fp + male_tn) > 0 else 0
    
    #qqual opportunity difference
    equal_opportunity_diff = male_tpr - female_tpr
    
    #statistical parity difference
    female_selection_rate = (female_tp + female_fp) / len(y_true[female_mask]) if len(y_true[female_mask]) > 0 else 0
    male_selection_rate = (male_tp + male_fp) / len(y_true[male_mask]) if len(y_true[male_mask]) > 0 else 0
    statistical_parity_diff = male_selection_rate - female_selection_rate
    
    #disparate impact
    disparate_impact = female_selection_rate / male_selection_rate if male_selection_rate > 0 else float('inf')
    
    metrics = {
        'female_tpr': float(female_tpr),
        'male_tpr': float(male_tpr),
        'female_fpr': float(female_fpr),
        'male_fpr': float(male_fpr),
        'equal_opportunity_diff': float(equal_opportunity_diff),
        'statistical_parity_diff': float(statistical_parity_diff),
        'disparate_impact': float(disparate_impact)
    }
    
    return metrics