
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_adult_dataset, preprocess_data
from bias_analyzer import analyze_gender_bias
from augmenter import GenderBiasAugmenter
from model import train_model, evaluate_model
from evaluator import compare_models, plot_fairness_metrics, plot_gender_income_bias
import config

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    #load and preprocess the Adult Income dataset
    df = load_adult_dataset()
    X, y, feature_names = preprocess_data(df)
    
    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    print("Analyzing initial gender bias...")
    #analyze initial gender bias in the dataset
    gender_idx = feature_names.index('gender_Female')
    bias_report = analyze_gender_bias(X_train, y_train, gender_idx)
    print(bias_report)
    
    #visualize the gender bias in income distribution
    # print("\nVisualizing initial gender bias in income distribution...")
    # plot_gender_income_bias(X_train, y_train, gender_idx)
    
    print("Training baseline model without augmentation...")
    #train baseline model without augmentation
    baseline_model = train_model(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, gender_idx)
    
    print("Training model with dynamic gender bias augmentation...")
    #initialize the gender bias augmenter
    augmenter = GenderBiasAugmenter(
        augmentation_rate=config.AUGMENTATION_RATE,
        bias_threshold=config.BIAS_THRESHOLD
    )
    
    #train model with dynamic augmentation
    augmented_model = train_model(
        X_train, y_train, 
        augmenter=augmenter, 
        gender_idx=gender_idx
    )
    augmented_metrics = evaluate_model(augmented_model, X_test, y_test, gender_idx)
    
    print("Evaluating results...")
    #compare model performances
    comparison = compare_models(baseline_metrics, augmented_metrics)
    print("\nModel Comparison:")
    print(comparison)
    
    #visualize results
    plot_fairness_metrics(baseline_metrics, augmented_metrics)
    
    print("\nExperiment complete. Dynamic dataset augmentation impact demonstrated.")