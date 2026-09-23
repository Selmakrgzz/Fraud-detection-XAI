import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay

# load and merge data
train_tx = pd.read_csv('data/train/train_transaction.csv')
train_id = pd.read_csv('data/train/train_identity.csv')
train = train_tx.merge(train_id, on='TransactionID', how='left')

y = train['isFraud']
X = train.drop(columns=['isFraud', 'TransactionID'])

# make categorical cols explicit
categorical_cols = X.select_dtypes(include=['object']).columns
X[categorical_cols] = X[categorical_cols].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# calculating the scale positive weight
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

# create model instance
bst = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    objective='binary:logistic',
    enable_categorical=True,
    scale_pos_weight= scale_pos_weight,
    eval_metric='aucpr'
)

# fit model
bst.fit(X_train, y_train)

# make predictions
preds = bst.predict(X_test)

#
#
#
'''''''''Report and plots'''''''''

# how many it predicted as fraud
print("Predicted fraud count:", np.sum(preds == 1), "out of", len(preds))
print("Actual fraud count:   ", np.sum(y_test == 1), "out of", len(y_test))

# precision/recall/f1 per class
print(classification_report(y_test, preds, target_names=['not fraud', 'fraud']))

# probability based metrics
probs = bst.predict_proba(X_test)[:, 1]
print("PR-AUC:", average_precision_score(y_test, probs))
print("ROC-AUC:", roc_auc_score(y_test, probs))

# roc_curve computes the FPR/TPR pairs at every threshold automatically
fpr, tpr, thresholds = roc_curve(y_test, probs)

RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
plt.title('ROC Curve - Fraud Detection')
plt.legend()
plt.show()