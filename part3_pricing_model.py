from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, roc_auc_score,roc_curve


import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from itertools import product

import pickle
import numpy as np


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel(nn.Module):
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False, dropout=0):
        super(PricingModel, self).__init__()
        self.fc1 = nn.Linear(in_features=36, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=1)



        self.drop_layer = nn.Dropout(p=dropout)



        self.y_mean = None
        self.base_classifier = None
        self.calibrate = calibrate_probabilities
        self.trained = False
        self.scaler = None
        self.encoder = None
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
    """

    def forward(self, t):
        t = F.leaky_relu(self.fc1(t)) # activiation hidden layer 1
        t = F.leaky_relu(self.fc2(t))
        t = self.drop_layer(t)# activation hidden layer 2

        t = F.leaky_relu(self.fc3(t))

        t = torch.sigmoid(self.out(t))
        return t
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """

        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """


        # =============================================================
        # # PART2 FORM

        del_ = [0, 8, 11, 13, 20, 21, 27, 28, 29, 30, 31, 32, 33, 34] #drop variables
        X_raw = X_raw.drop(X_raw.columns[del_],axis=1)
        index_cat = [1, 4, 5, 6, 7, 9, 15, 19]  #get indexes of categorical variables
        #print(index_cat.shape)

        sub_set = X_raw.iloc[:,index_cat] #get categorical variables


        if not self.trained:
            self.encoder = OneHotEncoder()
            self.encoder.fit(sub_set)

        sub_set = self.encoder.transform(sub_set)
        sub_set = pd.DataFrame(data=sub_set.toarray())
        X_raw = X_raw.drop(X_raw.columns[index_cat],axis=1)
        X_raw = pd.concat([X_raw.reset_index(drop=True),sub_set.reset_index(drop=True)],axis=1)

        X=X_raw.to_numpy()

        index_con = sorted(set(range(index_cat[0], index_cat[-1])) - set(index_cat))
        index_con.insert(0,0)
        X_con = X[:,index_con]
        col_mean = np.nanmean(X_con, axis=0)
        inds = np.where(np.isnan(X_con))
        X_con[inds] = np.take(col_mean, inds[1])

        #get replace missing values with average of columns
        if not self.trained:
            self.scaler = MinMaxScaler().fit(X_con)

        X_con = self.scaler.transform(X_con)
        X[:, index_con] = X_con


        return X






    def fit(self, X_raw, y_raw, claims_raw,epochs=100,batch_size=100,lr=0.01):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """

        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        y_raw = y_raw.to_numpy()
        y_claim = claims_raw.to_numpy()

        nnz = np.where(y_claim != 0)[0]
        self.y_mean=np.mean(y_claim[nnz])

        X_clean = self._preprocessor(X_raw)

        class_count = np.unique(y_raw, return_counts=True)[1]

        weight = 1. / class_count
        samples_weight = torch.as_tensor([weight[j] for j in y_raw.astype(int)])

        sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))

        X = torch.as_tensor(X_clean, dtype=torch.float32)
        y = torch.as_tensor(y_raw, dtype=torch.float32)

        train_data = data.TensorDataset(X, y)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)

        criterion = torch.nn.BCELoss()
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                X, y = batch

                optimiser.zero_grad()  # zero-out existing grad
                preds = self(X).squeeze()


                loss = criterion(preds, y)


                loss.backward()  # calculate gradients
                optimiser.step()  # gradient descent



        self.trained = True
        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_clean, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        # return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        X_clean = self._preprocessor(X_raw)
        X = torch.as_tensor(X_clean, dtype=torch.float32)
        y_pred = self(X).squeeze().detach().numpy()
        return y_pred

    def evaluate_architecture(self, y_true, y_pred, threshold = 0.5):
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

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean *0.90

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
        return trained_model


def ClaimClassifierHyperParameterSearch(X_train, y_train, X_val, y_val, params, Y_amount):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    param_values = [v for v in params.values()]
    best_performance = None
    best_auc = 0
    best_params = None

    for batch_size, lr, threshold, dropout in product(*param_values):
        print("\n************************************")
        print("TRAINING MODEL w/ Batch Size =", batch_size, ", Learning rate =", lr,
              ", Threshold =", threshold, ", Dropout =", dropout)

        model = PricingModel(dropout)
        model.fit(X_train, y_train, Y_amount, batch_size=batch_size, lr=lr)


        print("\nTRAINING")
        y_pred = model.predict_claim_probability(X_train)
        train_peformance = model.evaluate_architecture(y_train, y_pred, threshold)

        print("\nVALIDATION")
        y_pred = model.predict_claim_probability(X_val)
        performance = model.evaluate_architecture(y_val, y_pred, threshold)
        print("************************************")

        auc = performance['auc']
        if auc > best_auc:
            best_auc = auc
            best_params = dict(batch_size=batch_size, lr=lr, threshold=threshold, dropout=dropout)
            model.save_model()
            print("Saved model")
    #
    print("\n************************************")
    print("BEST MODEL AFTER GRID SEARCH")
    print("Best params:", best_params)
    print("Performance:", best_auc)

    return best_params
