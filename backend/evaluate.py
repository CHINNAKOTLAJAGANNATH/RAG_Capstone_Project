from sklearn.metrics import f1_score
import re

def compute_f1(pred, true):
    # Tokenize and clean text
    pred_tokens = set(re.findall(r'\w+', pred.lower()))
    true_tokens = set(re.findall(r'\w+', true.lower()))

    y_true = [1] * len(true_tokens)
    y_pred = [1 if token in pred_tokens else 0 for token in true_tokens]

    return f1_score(y_true, y_pred, zero_division=1)

# F1 = 2 × (Precision × Recall) / (Precision + Recall)