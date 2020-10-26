import numpy as np
import pickle
import sklearn
from sklearn.metrics import balanced_accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


class ClaimClassifier(nn.Module):

    def __init__(self, dropout):
        super(ClaimClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=9, out_features=36)
        self.fc2 = nn.Linear(in_features=36, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=9)
        self.drop_layer = nn.Dropout(p=dropout)
        self.out = nn.Linear(in_features=9, out_features=1)

        self.trained = False
        self.scaler = None


    def forward(self, t):
        t = F.leaky_relu(self.fc1(t))  # activiation hidden layer 1
        t = F.leaky_relu(self.fc2(t))  # activation hidden layer 2
        t = self.drop_layer(t)  # dropout neurons according to specified probability
        t = F.leaky_relu(self.fc3(t))  # activation hidden layer 3
        t = self.drop_layer(t)  # dropout neurons according to specified probability
        t = F.leaky_relu(self.fc4(t))  # activation hidden layer
        t = torch.sigmoid(self.out(t))

        return t

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # no missing values found
        # no inconsistent samples found
        # only numerical values

        # only use training scalar
        if not self.trained:
            self.scaler = StandardScaler().fit(X_raw.to_numpy())

        # return scaled data according to training scalar
        return self.scaler.transform(X_raw.to_numpy())

    def fit(self, X_raw, y_raw, batch_size=50, lr=0.0001, epochs=100,
            weighted_sampling = False, weighted_loss = False, weight_decay=0):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        X_clean = self._preprocessor(X_raw)

        # deal with imbalanced data according to parameter settings
        class_count = np.unique(y_raw, return_counts=True)[1]
        weight = 1. / class_count
        if weighted_sampling:
            samples_weight = np.array([weight[y] for y in y_raw])
        else:
            samples_weight = np.ones(len(y_raw))  # don't apply weights

        sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))

        # load the data for pytorch compatability
        X = torch.as_tensor(X_clean, dtype=torch.float32)
        y = torch.as_tensor(y_raw.to_numpy(), dtype=torch.float32)
        train_data = data.TensorDataset(X, y)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=1, sampler=sampler)

        # perform mini-batch gradient descent
        optimiser = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                X, y = batch
                optimiser.zero_grad()  # zero-out existing grad
                preds = self(X).squeeze()

                if weighted_loss:
                    loss_weight = torch.as_tensor([weight[sample] for sample in y.numpy().astype(int)])
                    criterion = nn.BCELoss(weight=loss_weight)
                else:
                    criterion = nn.BCELoss()

                loss = criterion(preds, y)
                loss.backward()  # calculate gradients
                optimiser.step()  # gradient descent

            #print("Loss:", loss.item(), "in epoch", epoch)

        self.trained = True
        pass

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # first clean the data before
        X_clean = self._preprocessor(X_raw)
        X = torch.as_tensor(X_clean, dtype=torch.float32)

        # run cleaned data through the model to get predictions
        y_pred = self(X).squeeze().detach().numpy()
        return y_pred

    def evaluate_architecture(self, y_true, y_pred, threshold=0.5):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # convert prediction probs to binary classes according to threshold
        y_pred = np.where(y_pred > threshold, 1, 0)

        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        performance = dict(balanced_acc=balanced_acc, avg_precision=avg_precision, f1 = f1, auc = auc)

        print("Balanced Accuracy Score:", balanced_acc)
        print("Average Precision Score:", avg_precision)
        print("F1 Score:", f1)
        print("AUC of ROC Score:", auc)

        return performance

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(X_train, y_train, X_val, y_val, params):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    param_values = [v for v in params.values()]
    best_params = None
    best_performance = None
    best_f1 = 0

    # grid search over all hyperameter values
    for batch_size, lr, threshold, epochs, \
        weighted_sampling, weighted_loss, \
        weight_decay, dropout in product(*param_values):
        print("\n************************************")
        print("TRAINING MODEL w/ Batch Size =", batch_size, ", Learning rate =", lr,
              ", Threshold =", threshold, ", Epochs = ", epochs,
              ", Weighted samples =", weighted_sampling, ", Loss weighted =", weighted_loss,
              ", Weight decay =", weight_decay, ", Dropout =", dropout)

        model = ClaimClassifier(dropout=dropout)
        model.fit(X_train, y_train,
                  batch_size=batch_size, lr=lr, epochs=epochs,
                  weighted_sampling=weighted_sampling, weighted_loss=weighted_loss,
                  weight_decay=weight_decay)

        # see if overfitting/underfitting
        print("\nMODEL TRAINING PERFORMANCE")
        y_pred_train = model.predict(X_train)
        training_performance = model.evaluate_architecture(y_train, y_pred_train, threshold)
        print("\nMODEL VALIDATION PERFORMANCE")
        y_pred_val = model.predict(X_val)
        performance = model.evaluate_architecture(y_val, y_pred_val, threshold)
        f1 = performance['f1']
        print("************************************")

        if f1 > best_f1:  # use validation data f1 as comparison
            best_performance = performance
            best_f1 = f1
            best_params = dict(batch_size=batch_size, lr=lr, threshold=threshold, epochs=epochs,
                               weighted_sampling=weighted_sampling, weighted_loss=weighted_loss,
                               weight_decay=weight_decay, dropout=dropout)
            model.save_model()  # save each best model

    print("\n************************************")
    print("BEST MODEL AFTER GRID SEARCH")
    print("Best params:", best_params)
    print("Performance:", best_performance)
    return best_params

