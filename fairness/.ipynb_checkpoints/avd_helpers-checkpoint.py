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


##################### LFR ########################################

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d


def check_inputs(X, y, sample_weight=None, ensure_2d=True):
    """Input validation for debiasing algorithms.
    Checks all inputs for consistent length, validates shapes (optional for X),
    and returns an array of all ones if sample_weight is ``None``.
    Args:
        X (array-like): Input data.
        y (array-like, shape = (n_samples,)): Target values.
        sample_weight (array-like, optional): Sample weights.
        ensure_2d (bool, optional): Whether to raise a ValueError if X is not
            2D.
    Returns:
        tuple:
            * **X** (`array-like`) -- Validated X. Unchanged.
            * **y** (`array-like`) -- Validated y. Possibly converted to 1D if
              not a :class:`pandas.Series`.
            * **sample_weight** (`array-like`) -- Validated sample_weight. If no
              sample_weight is provided, returns a consistent-length array of
              ones.
    """
    if ensure_2d and X.ndim != 2:
        raise ValueError("Expected X to be 2D, got ndim == {} instead.".format(
                X.ndim))
    if not isinstance(y, pd.Series):  # don't cast Series -> ndarray
        y = column_or_1d(y)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])
    check_consistent_length(X, y, sample_weight)
    return X, y, sample_weight

def check_groups(arr, prot_attr, ensure_binary=False):
    """Get groups from the index of arr.
    If there are multiple protected attributes provided, the index is flattened
    to be a 1-D Index of tuples. If ensure_binary is ``True``, raises a
    ValueError if there are not exactly two unique groups. Also checks that all
    provided protected attributes are in the index.
    Args:
        arr (array-like): Either a Pandas object containing protected attribute
            information in the index or array-like with explicit protected
            attribute array(s) for `prot_attr`.
        prot_attr (label or array-like or list of labels/arrays): Protected
            attribute(s). If contains labels, arr must include these in its
            index. If ``None``, all protected attributes in ``arr.index`` are
            used. Can also be 1D array-like of the same length as arr or a
            list of a combination of such arrays and labels in which case, arr
            may not necessarily be a Pandas type.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.
    Returns:
        tuple:
            * **groups** (:class:`pandas.Index`) -- Label (or tuple of labels)
              of protected attribute for each sample in arr.
            * **prot_attr** (`FrozenList`) -- Modified input. If input is a
              single label, returns single-item list. If input is ``None``
              returns list of all protected attributes.
    """
    arr_is_pandas = isinstance(arr, (pd.DataFrame, pd.Series))
    if prot_attr is None:  # use all protected attributes provided in arr
        if not arr_is_pandas:
            raise TypeError("Expected `Series` or `DataFrame` for arr, got "
                           f"{type(arr).__name__} instead. Otherwise, pass "
                            "explicit prot_attr array(s).")
        groups = arr.index
    elif arr_is_pandas:
        df = arr.index.to_frame()
        groups = df.set_index(prot_attr).index  # let pandas handle errors
    else:  # arr isn't pandas. might be okay if prot_attr is array-like
        df = pd.DataFrame(index=[None]*len(arr))  # dummy to check lengths match
        try:
            groups = df.set_index(prot_attr).index
        except KeyError as e:
            raise TypeError("arr does not include protected attributes in the "
                            "index. Check if this got dropped or prot_attr is "
                            "formatted incorrectly.") from e
    prot_attr = groups.names
    groups = groups.to_flat_index()

    n_unique = groups.nunique()
    if ensure_binary and n_unique != 2:
        raise ValueError("Expected 2 protected attribute groups, got "
                        f"{groups.unique() if n_unique > 5 else n_unique}")

    return groups, prot_attr

class LearnedFairRepresentations(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Learned Fair Representations.
    Learned fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [#zemel13]_. It can also be used as an in-
    processing method by utilizing the learned target coefficients.
    References:
        .. [#zemel13] `R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
           "Learning Fair Representations." International Conference on Machine
           Learning, 2013. <http://proceedings.mlr.press/v28/zemel13.html>`_
    # Based on code from https://github.com/zjelveh/learning-fair-representations
    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            reweighing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the transformer.
        priv_group_ (scalar): The label of the privileged group.
        coef_ (array, shape (n_prototypes, 1) or (n_prototypes, n_classes)):
            Coefficient of the intermediate representation for classification.
        prototypes_ (array, shape (n_prototypes, n_features)): The prototype set
            used to form a probabilistic mapping to the intermediate
            representation. These act as clusters and are in the same space as
            the samples.
        n_iter_ (int): Actual number of iterations.
    """

    def __init__(self, prot_attr=None, n_prototypes=5, reconstruct_weight=0.01,
                 target_weight=1., fairness_weight=50., tol=1e-4, max_iter=200,
                 verbose=0, random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the reweighing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            n_prototypes (int, optional): Size of the set of "prototypes," Z.
            reconstruct_weight (float, optional): Weight coefficient on the L_x
                loss term, A_x.
            target_weight (float, optional): Weight coefficient on the L_y loss
                term, A_y.
            fairness_weight (float, optional): Weight coefficient on the L_z
                loss term, A_z.
            tol (float, optional): Tolerance for stopping criteria.
            max_iter (int, optional): Maximum number of iterations taken for the
                solver to converge.
            verbose (int, optional): Verbosity. 0 = silent, 1 = final loss only,
                2 = print loss every 50 iterations.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """
        self.prot_attr = prot_attr
        self.n_prototypes = n_prototypes
        self.reconstruct_weight = reconstruct_weight
        self.target_weight = target_weight
        self.fairness_weight = fairness_weight
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, priv_group=1, sample_weight=None):
        """Compute the transformation parameters that lead to fair
        representations.
        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
            priv_group (scalar, optional): The label of the privileged group.
            sample_weight (array-like, optional): Sample weights.
        Returns:
            self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        rng = check_random_state(self.random_state)

        groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        priv = (groups == priv_group)
        self.priv_group_ = priv_group
        self.groups_ = np.unique(groups)

        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)
        if n_classes == 2:
            n_classes = 1  # XXX
        n_feat = X.shape[1]
        w_size = self.n_prototypes*n_classes

        i = 0
        eps = np.finfo(np.float64).eps

        def LFR_optim_objective(x, X, y, priv):
            nonlocal i
            x = torch.as_tensor(x).requires_grad_()
            w = x[:w_size].view(-1, n_classes)
            v = x[w_size:].view(-1, n_feat)

            M = torch.softmax(-torch.cdist(X, v), dim=1)
            y_pred = M.matmul(w).squeeze(1)

            L_x = F.mse_loss(M.matmul(v), X)
            L_y = F.cross_entropy(y_pred, y) if n_classes > 1 else \
                  F.binary_cross_entropy(y_pred.clamp(eps, 1-eps), y.type_as(w))
            L_z = F.l1_loss(torch.mean(M[priv], 0), torch.mean(M[~priv], 0))
            loss = (self.reconstruct_weight * L_x + self.target_weight * L_y
                  + self.fairness_weight * L_z)

            loss.backward()
            if self.verbose > 1 and i % 50 == 0:
                print("iter: {:{}d}, loss: {:7.3f}, A_x*L_x: {:7.3f}, A_y*L_y: "
                      "{:7.3f}, A_z*L_z: {:7.3f}".format(i,
                        int(np.log10(self.max_iter)+1), loss,
                        self.reconstruct_weight*L_x, self.target_weight*L_y,
                        self.fairness_weight*L_z))
            i += 1
            return loss.item(), x.grad.numpy()

        x0 = rng.random(w_size + self.n_prototypes*n_feat)
        bounds = [(0, 1)]*w_size + [(None, None)]*self.n_prototypes*n_feat
        res = s_optim.minimize(LFR_optim_objective, x0=x0, method='L-BFGS-B',
                args=(torch.tensor(X.to_numpy()), torch.as_tensor(y), priv),
                jac=True, bounds=bounds, options={'gtol': self.tol,
                'maxiter': self.max_iter})

        self.coef_ = res.x[:w_size].reshape(-1, n_classes)
        self.prototypes_ = res.x[w_size:].reshape(-1, n_feat)
        self.n_iter_ = res.nit

        if res.status == 0 and self.verbose:
            print("Converged! iter: {}, loss: {:.3f}".format(res.nit, res.fun))
        elif res.status == 1:
            warnings.warn('lbfgs failed to converge. Increase the number of '
                          'iterations.', ConvergenceWarning)
        elif res.status == 2:
            warnings.warn('lbfgs failed to converge: {}'.format(
                          res.message.decode()), ConvergenceWarning)
        return self

    def transform(self, X):
        """Transform the dataset using the learned model parameters.
        Args:
            X (pandas.DataFrame): Training samples.
        Returns:
            pandas.DataFrame: Transformed samples.
        """
        M = softmax(-cdist(X, self.prototypes_), axis=1)
        Xt = M.dot(self.prototypes_)
        return pd.DataFrame(Xt, columns=X.columns, index=X.index)

    def predict_proba(self, X):
        """Transform the targets using the learned model parameters.
        Args:
            X (pandas.DataFrame): Training samples.
        Returns:
            numpy.ndarray: Transformed targets. Returns the probability of the
            sample for each class in the model, where classes are ordered as
            they are in ``self.classes_``.
        """
        M = softmax(-cdist(X, self.prototypes_), axis=1)
        yt = M.dot(self.coef_)
        if yt.shape[1] == 1:
            yt = np.c_[1-yt, yt]
        else:
            yt = softmax(yt, axis=1)
        return yt

    def predict(self, X):
        """Transform the targets using the learned model parameters.
        Args:
            X (pandas.DataFrame): Training samples.
        Returns:
            numpy.ndarray: Transformed targets.
        """
        probas = self.predict_proba(X)
        return self.classes_[probas.argmax(axis=1)]