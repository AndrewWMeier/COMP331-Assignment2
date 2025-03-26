

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Showing the inital dataset biases
def plot_gender_income_bias(X, y, gender_idx):
    """
    Create a simple plot showing gender bias in income distribution
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Income labels (0 for low, 1 for high)
    gender_idx : int
        Index of gender feature (1 for female, 0 for male)
    """
    
    # Extract gender
    gender = X[:, gender_idx]
    
    # Create DataFrame for easy analysis
    data = pd.DataFrame({
        'Gender': ['Female' if g == 1 else 'Male' for g in gender],
        'Income': ['High Income' if income == 1 else 'Low Income' for income in y]
    })
    
    # Calculate percentages for plotting
    income_by_gender = pd.crosstab(data['Gender'], data['Income'], normalize='index') * 100
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    income_by_gender.plot(kind='bar', color=['lightgray', 'darkblue'])
    
    plt.title('Income Distribution by Gender', fontsize=14)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add percentage labels on bars
    for i, gender in enumerate(income_by_gender.index):
        for j, col in enumerate(income_by_gender.columns):
            plt.text(i, income_by_gender.iloc[i, j] + 1, 
                    f'{income_by_gender.iloc[i, j]:.1f}%', 
                    ha='center')
    
    plt.legend(title='Income Level')
    plt.tight_layout()
    plt.savefig('gender_income_bias.png')
    plt.show()
    
    # Print the actual numbers
    print("\nIncome Distribution by Gender:")
    print(income_by_gender)
    
    # Calculate and print the bias metrics
    female_high = np.sum((gender == 1) & (y == 1))
    female_total = np.sum(gender == 1)
    female_high_rate = female_high / female_total if female_total > 0 else 0
    
    male_high = np.sum((gender == 0) & (y == 1))
    male_total = np.sum(gender == 0)
    male_high_rate = male_high / male_total if male_total > 0 else 0
    
    bias_gap = male_high_rate - female_high_rate
    
    print(f"\nHigh Income Rate - Male: {male_high_rate:.2%}")
    print(f"High Income Rate - Female: {female_high_rate:.2%}")
    print(f"Gender Bias Gap: {bias_gap:.2%}")

def compare_models(baseline_metrics, augmented_metrics):
    """
    Compare baseline and augmented model performances
    
    Parameters:
    -----------
    baseline_metrics : dict
        Metrics from baseline model
    augmented_metrics : dict
        Metrics from augmented model
        
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': [
            'Accuracy', 
            'F1 Score', 
            'AUC',
            'Female TPR', 
            'Male TPR', 
            'Equal Opportunity Difference',
            'Statistical Parity Difference',
            'Disparate Impact'
        ],
        'Baseline': [
            baseline_metrics['accuracy'],
            baseline_metrics['f1_score'],
            baseline_metrics['auc'],
            baseline_metrics['bias_metrics']['female_tpr'],
            baseline_metrics['bias_metrics']['male_tpr'],
            baseline_metrics['bias_metrics']['equal_opportunity_diff'],
            baseline_metrics['bias_metrics']['statistical_parity_diff'],
            baseline_metrics['bias_metrics']['disparate_impact']
        ],
        'Augmented': [
            augmented_metrics['accuracy'],
            augmented_metrics['f1_score'],
            augmented_metrics['auc'],
            augmented_metrics['bias_metrics']['female_tpr'],
            augmented_metrics['bias_metrics']['male_tpr'],
            augmented_metrics['bias_metrics']['equal_opportunity_diff'],
            augmented_metrics['bias_metrics']['statistical_parity_diff'],
            augmented_metrics['bias_metrics']['disparate_impact']
        ]
    })
    
    # Calculate improvement
    comparison['Difference'] = comparison['Augmented'] - comparison['Baseline']
    comparison['Percent Change'] = ((comparison['Augmented'] - comparison['Baseline']) / 
                                   comparison['Baseline'] * 100).round(2)
    
    # Format numbers
    for col in ['Baseline', 'Augmented', 'Difference']:
        comparison[col] = comparison[col].round(4)
    
    return comparison

def plot_fairness_metrics(baseline_metrics, augmented_metrics):
    """
    Plot fairness metrics comparison
    
    Parameters:
    -----------
    baseline_metrics : dict
        Metrics from baseline model
    augmented_metrics : dict
        Metrics from augmented model
    """
    # Set up figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. True Positive Rate by Gender
    tpr_data = {
        'Gender': ['Female', 'Female', 'Male', 'Male'],
        'Model': ['Baseline', 'Augmented', 'Baseline', 'Augmented'],
        'TPR': [
            baseline_metrics['bias_metrics']['female_tpr'],
            augmented_metrics['bias_metrics']['female_tpr'],
            baseline_metrics['bias_metrics']['male_tpr'],
            augmented_metrics['bias_metrics']['male_tpr']
        ]
    }
    tpr_df = pd.DataFrame(tpr_data)
    
    sns.barplot(x='Gender', y='TPR', hue='Model', data=tpr_df, ax=axs[0])
    axs[0].set_title('True Positive Rate by Gender')
    axs[0].set_ylim(0, 1)
    
    # 2. Equal Opportunity Difference
    eod_data = {
        'Model': ['Baseline', 'Augmented'],
        'Equal Opportunity Difference': [
            baseline_metrics['bias_metrics']['equal_opportunity_diff'],
            augmented_metrics['bias_metrics']['equal_opportunity_diff']
        ]
    }
    eod_df = pd.DataFrame(eod_data)
    
    sns.barplot(x='Model', y='Equal Opportunity Difference', data=eod_df, ax=axs[1])
    axs[1].set_title('Equal Opportunity Difference\n(Closer to 0 is better)')
    axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # 3. Disparate Impact
    di_data = {
        'Model': ['Baseline', 'Augmented'],
        'Disparate Impact': [
            baseline_metrics['bias_metrics']['disparate_impact'],
            augmented_metrics['bias_metrics']['disparate_impact']
        ]
    }
    di_df = pd.DataFrame(di_data)
    
    sns.barplot(x='Model', y='Disparate Impact', data=di_df, ax=axs[2])
    axs[2].set_title('Disparate Impact\n(Closer to 1 is better)')
    axs[2].axhline(y=1, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fairness_metrics.png')
    plt.show()

