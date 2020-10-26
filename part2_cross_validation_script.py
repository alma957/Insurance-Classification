from part2_claim_classifier import *
from sklearn.model_selection import train_test_split
import pandas as pd

dat = pd.read_csv("part2_training_data.csv")
X = dat.drop(columns=["claim_amount", "made_claim"])
y = dat["made_claim"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

params = dict(batch_size=[100, 200], lr=[0.1, 0.01, 0.001], threshold=[0.5], epochs=[100],
              weighted_sampling=[True, False], weighted_loss=[True, False],
              weight_decay=[0], dropout=[0.9, 0.7, 0.5])

best_params = ClaimClassifierHyperParameterSearch(X_train, y_train, X_val, y_val, params)
threshold = best_params['threshold']

best_model = load_model()

y_pred = best_model.predict(X_test)
print("**********************************************")
print("Evaluating the best model on the test set...")
best_model.evaluate_architecture(y_test, y_pred, threshold=threshold)

