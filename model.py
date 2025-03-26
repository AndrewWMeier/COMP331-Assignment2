# model.py - ML model definition and training

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from bias_analyzer import compute_bias_metrics
import config

def train_model(X, y, augmenter=None, gender_idx=None, batch_size=config.BATCH_SIZE):
    """
    Train model with optional dynamic augmentation
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    augmenter : GenderBiasAugmenter, optional
        Augmenter for dynamic dataset augmentation
    gender_idx : int, optional
        Index of the gender_Female feature
    batch_size : int
        Size of batches for dynamic augmentation
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_SEED
    )
    
    # If no augmenter, simply fit the model
    if augmenter is None or gender_idx is None:
        model.fit(X, y)
        return model
    
    # Implement dynamic augmentation during training
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Create batches for augmentation
    n_batches = int(np.ceil(n_samples / batch_size))
    X_augmented = []
    y_augmented = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # Augment the batch
        X_batch_aug, y_batch_aug = augmenter.augment(X_batch, y_batch, gender_idx)
        
        X_augmented.append(X_batch_aug)
        y_augmented.append(y_batch_aug)
    
    # Combine all augmented batches
    X_final = np.vstack(X_augmented)
    y_final = np.concatenate(y_augmented)
    
    # Fit model on augmented data
    model.fit(X_final, y_final)
    
    return model

def evaluate_model(model, X, y, gender_idx):
    """
    Evaluate model performance and fairness
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        Target values
    gender_idx : int
        Index of the gender_Female feature
        
    Returns:
    --------
    dict
        Performance and fairness metrics
    """
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Extract gender
    gender = X[:, gender_idx]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    # Calculate bias metrics
    bias_metrics = compute_bias_metrics(y, y_pred, gender)
    
    # Combine all metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auc': float(auc),
        'bias_metrics': bias_metrics
    }
    
    return metrics