# def compute_f1(prediction, ground_truth):
#     prediction_tokens = prediction.lower().split()
#     ground_truth_tokens = ground_truth.lower().split()

#     common = set(prediction_tokens) & set(ground_truth_tokens)
#     if len(common) == 0:
#         return 0.0

#     precision = len(common) / len(prediction_tokens)
#     recall = len(common) / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


from sklearn.metrics import f1_score
import re

def compute_f1(pred, true):
    # Tokenize and clean text
    pred_tokens = set(re.findall(r'\w+', pred.lower()))
    true_tokens = set(re.findall(r'\w+', true.lower()))

    y_true = [1] * len(true_tokens)
    y_pred = [1 if token in pred_tokens else 0 for token in true_tokens]

    return f1_score(y_true, y_pred, zero_division=1)

# # F1 = 2 × (Precision × Recall) / (Precision + Recall)

# The F1-score is used to evaluate the quality of the generated answers by comparing them against the ground truth answers. 
# The process involves the following steps:

# 1. Tokenization and Cleaning:
# Both predicted and true answers are converted to lowercase.
# Non-word characters are removed using regex (\w+) to extract tokens.

# 2. Binary Matching:
# A token is marked as 1 in the prediction (y_pred) if it exists in the predicted answer.
# All ground truth tokens (y_true) are marked as 1.

# 3. F1-score Calculation:
# Precision: How many predicted tokens are correct?
# Recall: How many correct tokens were predicted?
# F1-score: Harmonic mean of precision and recall, computed using sklearn.metrics.f1_score.

# 4. Handling Edge Cases:
# If there are no predicted tokens in the true set, zero_division=1 ensures F1 doesn't throw an error and defaults to 1 (optimistic strategy).