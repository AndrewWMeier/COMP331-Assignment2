# augmenter.py - Dynamic dataset augmentation implementation for gender bias

import numpy as np
from sklearn.neighbors import NearestNeighbors
import config

class GenderBiasAugmenter:
    """
    Dynamically augments dataset to reduce gender bias during training
    """
    def __init__(self, augmentation_rate=0.5, bias_threshold=0.1):
        """
        Initialize the augmenter
        
        Parameters:
        -----------
        augmentation_rate : float
            Rate of augmentation (proportion of new samples to generate)
        bias_threshold : float
            Threshold for bias detection that triggers augmentation
        """
        self.augmentation_rate = augmentation_rate
        self.bias_threshold = bias_threshold
        self.neighbors_model = None
    
    def detect_bias(self, X, y, gender_idx):
        """
        Detect gender bias in current batch
        
        Returns:
        --------
        float
            Bias score
        """
        # Extract gender
        gender = X[:, gender_idx]
        
        # Count high-income samples by gender
        female_high_income = np.sum((gender == 1) & (y == 1))
        male_high_income = np.sum((gender == 0) & (y == 1))
        
        # Count total by gender
        female_count = np.sum(gender == 1)
        male_count = np.sum(gender == 0)
        
        # Calculate rates
        female_high_income_rate = female_high_income / female_count if female_count > 0 else 0
        male_high_income_rate = male_high_income / male_count if male_count > 0 else 0
        
        # Bias is the difference in high-income rates
        bias_score = male_high_income_rate - female_high_income_rate
        
        return bias_score
    
    def augment(self, X, y, gender_idx):
        """
        Dynamically augment the dataset based on detected bias
        
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
        tuple
            Augmented X and y
        """
        # Detect bias in current data
        bias_score = self.detect_bias(X, y, gender_idx)
        
        # If bias is below threshold, no augmentation needed
        if bias_score < self.bias_threshold:
            return X, y
        
        # Find female samples
        female_mask = X[:, gender_idx] == 1
        
        # Find high-income male samples
        high_income_male_mask = (X[:, gender_idx] == 0) & (y == 1)
        high_income_male_X = X[high_income_male_mask]
        
        # If no high-income males or no females, return original data
        if len(high_income_male_X) == 0 or np.sum(female_mask) == 0:
            return X, y
        
        # Initialize nearest neighbors model if not already done
        if self.neighbors_model is None:
            self.neighbors_model = NearestNeighbors(n_neighbors=min(5, len(X)))
            
        # Find female samples that are most similar to high-income males
        female_X = X[female_mask]
        self.neighbors_model.fit(female_X)
        
        # Determine number of samples to generate
        n_samples = int(len(high_income_male_X) * self.augmentation_rate)
        
        # Generate synthetic female samples with high income
        synthetic_X = []
        synthetic_y = []
        
        for _ in range(n_samples):
            # Randomly select a high-income male sample
            idx = np.random.randint(0, len(high_income_male_X))
            male_sample = high_income_male_X[idx]
            
            # Find nearest female samples
            distances, indices = self.neighbors_model.kneighbors([male_sample], n_neighbors=min(3, len(female_X)))
            
            # Select a random female neighbor
            female_idx = indices[0][np.random.randint(0, len(indices[0]))]
            female_sample = female_X[female_idx]
            
            # Create synthetic female sample (mixture of features except gender)
            synthetic_sample = male_sample.copy()
            
            # Keep gender as female
            synthetic_sample[gender_idx] = 1
            
            # Randomly choose features from either male or female sample
            for i in range(len(synthetic_sample)):
                if i != gender_idx:  # Preserve gender
                    # Randomly choose which sample to take feature from
                    if np.random.random() < 0.7:  # Bias towards high-income male features
                        synthetic_sample[i] = male_sample[i]
                    else:
                        synthetic_sample[i] = female_sample[i]
            
            synthetic_X.append(synthetic_sample)
            synthetic_y.append(1)  # High income
        
        # Combine original and synthetic data
        if len(synthetic_X) > 0:
            X_augmented = np.vstack([X, synthetic_X])
            y_augmented = np.concatenate([y, synthetic_y])
            return X_augmented, y_augmented
        else:
            return X, y