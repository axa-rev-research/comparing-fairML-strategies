import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.utils import check_random_state

import numpy as np
import pandas as pd
import scipy.optimize as s_optim
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from fairness.helpers import plot_distributions
from fairness.helpers import *
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

import warnings



torch.manual_seed(1)
np.random.seed(7)
#sns.set(style="white", palette="muted", color_codes=True, context="talk")
sns.set_theme()

############### REJECT OPTION CLASSIFICATION ###############

class RejectOptionClassifier:
    
    def __init__(self, theta):
        self.theta = theta
        
    def predict(self, X, Z):
        labels = (X>0.5)*1
        prot_labels = 1-Z
        
        m = np.max(np.array([[1-x for x in X], X]).T, axis=1)
        m[m<=self.theta] = prot_labels[m<=self.theta]
        m[m>self.theta] = labels[m>self.theta]
        return m


############## ADVERSARIAL LEARNING ##############

class Classifier_old(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier_old, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

    
def fit_clf(clf, clf_optimizer, clf_criterion, train_loader, num_epochs=40):
    losses = []
    for epoch in range(num_epochs):
        clf, loss_ = fit_one_epoch(clf, train_loader, clf_optimizer, clf_criterion)
        losses.append(loss_)
    return clf, losses

            
def fit_one_epoch(clf, data_loader, optimizer, criterion):
    loss_sum = 0
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y).mean()
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    return clf, loss_sum/len(data_loader)


class Adversary_old(nn.Module):

    def __init__(self, n_input=1, n_output=1, n_hidden=32):
        super(Adversary_old, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

def fit_adv(adv, clf, data_loader, adv_optimizer, adv_criterion, lambda_, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        adv, losss = fit_adv_one_epoch(adv, clf, data_loader, adv_optimizer, adv_criterion, lambda_)
        losses.append(losss)
    return adv, losses
    
    
def fit_adv_one_epoch(adv, clf, data_loader, optimizer, criterion, lambdas):
    loss_sum = 0
    for x, _, z in data_loader:
        p_y = clf.forward(x).detach()#.reshape(-1,32)
        adv.zero_grad()
        p_z = adv(p_y)
        loss = (criterion(p_z, z) * lambdas).mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return adv, loss_sum/len(data_loader)

def fit_clf_adv(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer, lambda_, num_epochs):
    losses = []
    for epoch in range(1, num_epochs):
        clf, adv, losss = fit_clf_adv_joint_one_epoch(clf, adv, data_loader, clf_criterion, adv_criterion,
                                clf_optimizer, adv_optimizer, lambda_)
        losses.append(losss)
    return clf, adv, losses

def fit_clf_adv_joint_one_epoch(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer, lambdas):
    
    # Train adversary
    for x, y, z in data_loader:
        p_y = clf.forward(x).detach()
        adv.zero_grad()
        p_z = adv(p_y)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()
 
    # Train classifier on single batch
    for x, y, z in data_loader:
        pass
    p_y = clf(x)
    p_y_adv = clf.forward(x).detach()
    p_z = adv(p_y_adv)
    clf.zero_grad()
    #p_z = adv(p_y)
    loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
    clf_loss = clf_criterion(p_y, y) - (adv_criterion(p_z, z) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()
    
    return clf, adv, clf_loss.item()

##################################################################
    
def split_scale_pipe(X, y, Z, stratify, test_size=0.5, random_state=11):
    (X_train, X_test, y_train, y_test,
     Z_train, Z_test) = train_test_split(X, y, Z, test_size=test_size,
                                         stratify=y, random_state=11)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                               columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler) 
    
    return (X_train, X_test, y_train, y_test, Z_train, Z_test)
