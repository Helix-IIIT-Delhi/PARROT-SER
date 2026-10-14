"""Evaluation Metrics for Speech Emotion Recognition"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_emotion_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute emotion recognition metrics.

    Args:
        predictions: Predicted emotion IDs
        targets: Ground truth emotion IDs

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for emotion recognition"""
    return confusion_matrix(targets, predictions)


def compute_per_emotion_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute metrics per emotion class"""
    return {
        "precision_per_class": precision_score(targets, predictions, average=None, zero_division=0),
        "recall_per_class": recall_score(targets, predictions, average=None, zero_division=0),
        "f1_per_class": f1_score(targets, predictions, average=None, zero_division=0),
    }
