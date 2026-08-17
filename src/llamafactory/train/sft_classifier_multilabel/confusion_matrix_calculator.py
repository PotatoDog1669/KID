import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_metrics(true_labels, pred_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

def main():
    # Load the CSV file (replace 'your_file.csv' with the actual file path)
    file_path = 'predictions_m7k_taskB_FocalLoss4.csv'
    df = pd.read_csv(file_path)
    
    categories = ['humour', 'sarcasm', 'offensive', 'motivational']
    metrics_dict = {cat: {} for cat in categories}
    all_true = []
    all_pred = []
    
    # Process each category
    for cat in categories:
        true_labels = df[f'label_{cat}'].values
        pred_labels = df[f'pred_label_{cat}'].values
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        
        metrics = calculate_metrics(true_labels, pred_labels)
        metrics_dict[cat] = metrics
    
    # Calculate average metrics
    avg_metrics = calculate_metrics(np.array(all_true), np.array(all_pred))
    
    # Print results
    for cat in categories:
        print(f"{cat.capitalize()}:")
        for metric, value in metrics_dict[cat].items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    print("Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()