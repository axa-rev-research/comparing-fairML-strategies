import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from fairlearn_int.metrics import equalized_odds_difference, equalized_odds_ratio
import matplotlib.patches as mpatches


def tsne_plot(df, methods, colors_map):
    """
    df: tsne axes
    methods: dict {method_name: df_changed_by_method}
    colors_map: dict{method_name: color}
    """
    
    plot_dfs = {}
    for m in methods.keys():
        plot_dfs[m] = df.loc[methods[m].index]
        
    fig, axs = plt.subplots(1, 1, figsize = (10,6))
    
    sns.kdeplot(data=df, x='c1', y='c2', ax=axs, label='All Instances', fill=True, 
            thresh=0.05, alpha=0.4, legend=True, color='Grey', cut=0.5)
    
    patch_all = mpatches.Patch(
        color='Grey', label='All Instances'
    )
    
    patches=[patch_all]
    
    for m in plot_dfs.keys():
        sns.kdeplot(data=plot_dfs[m], x='c1', y='c2', ax=axs, label='ROC', fill=False, thresh=0.4, color=colors_map[m])
        patch = mpatches.Patch(color=colors_map[m], label=m)
        patches.append(patch)
        
    axs.legend(handles=patches)
    axs.set_title('T-SNE projection of Datapoints')
    plt.tight_layout()


def disparate_mistreatment(data, sensitive_attribute, priv_value=1):
    tn_priv = len(data[(data.target==0) & (data[sensitive_attribute]==priv_value)])
    tn_disadv = len(data[(data.target==0) & (data[sensitive_attribute]!=priv_value)])

    tp_priv = len(data[(data.target==1) & (data[sensitive_attribute]==priv_value)])
    tp_disadv = len(data[(data.target==1) & (data[sensitive_attribute]!=priv_value)])
    
    fpr_priv, tpr_priv = metrics.confusion_matrix(
        data[data[sensitive_attribute]==priv_value].target,
        data[data[sensitive_attribute]==priv_value].fair_label
    )[[0,1],1]
    
    fpr_disadv, tpr_disadv = metrics.confusion_matrix(
        data[data[sensitive_attribute]!=priv_value].target,
        data[data[sensitive_attribute]!=priv_value].fair_label
    )[[0,1],1]
    
    #return fpr_disadv#, fpr_priv
    return (abs(fpr_priv/tn_priv*100 - fpr_disadv/tn_disadv*100),
        abs(tpr_priv/tp_priv*100 - tpr_disadv/tp_disadv*100))


def load_adult(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                 )
    
    input_data = input_data[~input_data.duplicated()]
    sensitive_attribs = ['sex']
    
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(sex=lambda df: (df['sex'] == 'Male').astype(int)))
    
    Z = Z.loc[:, ['sex']]
    
    y = (input_data['target'] == '>50K').astype(int)

    drop_columns = ['target', 'fnlwgt', 'sex']
    
    X = (input_data
         .drop(columns=drop_columns)
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    return X, y, Z


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


def plot_distributions(y_true, Z_true, y_pred, Z_pred=None, epoch=None, sex=True, race=True):
    
    subplot_df = Z_true.assign(y_pred=y_pred)
    num_subplots = 0
    if sex:
        subplot_df = subplot_df.assign(sex=lambda x: x['sex'].map({1: 'male', 0: 'female'}))
        num_subplots += 1
    if race:
        subplot_df = subplot_df.assign(race=lambda x: x['race'].map({1: 'white', 0: 'poc'}))
        num_subplots += 1
    
    fig, axes = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 4), sharey=True)
    
    if race and sex:
        _subplot(subplot_df, 'race', ax=axes[0])
        _subplot(subplot_df, 'sex', ax=axes[1])
    elif race:
        _subplot(subplot_df, 'race', ax=axes)
    elif sex:
        _subplot(subplot_df, 'sex', ax=axes)
        
    _performance_text(fig, y_true, Z_true, y_pred, Z_pred, epoch, race, sex)
    fig.tight_layout()
    return fig


def _subplot(subplot_df, col, ax):
    for label, df in subplot_df.groupby(col):
        sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)
    ax.set_title(f'Sensitive attribute: {col}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(col))


def _performance_text(fig, y_test, Z_test, y_pred, Z_pred=None, epoch=None, race=True, sex=True):

    if epoch is not None:
        fig.text(1.0, 0.9, f"Training epoch #{epoch}", fontsize='16')

    clf_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
    
    p_rules = {}
    if sex:
        p_rules['sex'] = p_rule(y_pred, Z_test['sex'])
    if race:
        p_rules['race'] = p_rule(y_pred, Z_test['race'])
    
    fig.text(1.0, 0.65, '\n'.join(["Classifier performance:",
                                   f"- ROC AUC: {clf_roc_auc:.2f}",
                                   f"- Accuracy: {clf_accuracy:.1f}"]),
             fontsize='16')
    fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                 [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                  for attr in p_rules.keys()]),
             fontsize='16')
    if Z_pred is not None:
        adv_roc_auc = metrics.roc_auc_score(Z_test, Z_pred)
        fig.text(1.0, 0.20, '\n'.join(["Adversary performance:",
                                       f"- ROC AUC: {adv_roc_auc:.2f}"]),
                 fontsize='16')
        
    
def pretrain_classifier(clf, data_loader, optimizer, criterion):
    loss_sum = 0
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    return clf, loss_sum/len(data_loader)

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch


def plot_decision_boundary(dataset, labels, model, axs, steps=1000, color_map='ocean_r'):
    color_map = plt.get_cmap(color_map)
    # Define region of interest by data limits
    xmin, xmax = dataset.iloc[:, 0].min() - 1, dataset.iloc[:, 0].max() + 1
    ymin, ymax = dataset.iloc[:, 1].min() - 1, dataset.iloc[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    model.eval()
    labels_predicted = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))

    # Plot decision boundary in region of interest
    labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    z = np.array(labels_predicted).reshape(xx.shape)
    
    # Get predicted labels on training data and plot
    train_labels_predicted = model(PandasDataSet()._df_to_tensor(dataset))
    axs.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=labels, cmap='autumn', lw=0) #.reshape(labels.size()[0])
    
    #fig, ax = plt.subplots()
    axs.contourf(xx, yy, z, cmap='Oranges', alpha=0.5)
    #plt.show()
    #return fig, ax
    return axs

# CHANGE


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion, lambdas):
    loss_sum = 0
    for x, _, z in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        p_z = adv(p_y)
        loss = (criterion(p_z, z) * lambdas).mean()
        #print(loss)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    return adv, loss_sum/len(data_loader)

def train(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer, lambdas):
    
    # Train adversary
    for x, y, z in data_loader:
        p_y = clf(x)
        adv.zero_grad()
        p_z = adv(p_y)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()
 
    # Train classifier on single batch
    for x, y, z in data_loader:
        pass
    p_y = clf(x)
    p_z = adv(p_y)
    clf.zero_grad()
    p_z = adv(p_y)
    loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
    clf_loss = clf_criterion(p_y, y) - (adv_criterion(adv(p_y), z) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()
    
    return clf, adv

def get_metrics(results_dfs, approaches, sensitive_feature, include_biased=True):
    accs = []
    bal_accs = []
    class_0_accs = []
    class_1_accs = []
    prules = []
    tpr_diffs = []
    fpr_diffs = []
    
    for df in results_dfs:
        accs.append(metrics.accuracy_score(df.target, df.fair_label))
        bal_accs.append(metrics.balanced_accuracy_score(df.target, df.fair_label))
        class_0_accs.append(metrics.accuracy_score(df[df[sensitive_feature]==0].target, df[df[sensitive_feature]==0].fair_label))
        class_1_accs.append(metrics.accuracy_score(df[df[sensitive_feature]==1].target, df[df[sensitive_feature]==1].fair_label))
        prules.append(p_rule(df.fair_label, df[sensitive_feature]))
        fpr_diff, tpr_diff = disparate_mistreatment(
            df, sensitive_feature
        )
        fpr_diffs.append(fpr_diff)
        tpr_diffs.append(tpr_diff)
        
    if include_biased:
        df = results_dfs[0]
        accs.append(metrics.accuracy_score(df.target, df.biased_label))
        bal_accs.append(metrics.balanced_accuracy_score(df.target, df.biased_label))
        class_0_accs.append(metrics.accuracy_score(df[df[sensitive_feature]==0].target, df[df[sensitive_feature]==0].biased_label))
        class_1_accs.append(metrics.accuracy_score(df[df[sensitive_feature]==1].target, df[df[sensitive_feature]==1].biased_label))
        prules.append(p_rule(df.biased_label, df[sensitive_feature]))
        fpr_diff, tpr_diff = disparate_mistreatment(
            df[[sensitive_feature, 'target', 'biased_label']].rename({'biased_label':'fair_label'}, axis=1), sensitive_feature
        )
        fpr_diffs.append(fpr_diff)
        tpr_diffs.append(tpr_diff)

        approaches = approaches + ['biased']
    
    return pd.DataFrame({'approach' : approaches, 'accuracy': accs, 'balanced accuracy': bal_accs, 
                         'class 0 accuracy': class_0_accs, 'class 1 accuracy': class_1_accs,
                         'p% rules': prules, 'fpr_diffs': fpr_diffs, 'tpr_diffs': tpr_diffs})

def get_categorical_counts(df, columns):
    dfs = []
    dfs_s = []
    
    for c in columns:
        c_counts = df.groupby(c).count()[['age']].reset_index().sort_values(by=['age'], ascending=False)\
                     .rename({'age':'count', c:'value'}, axis=1).assign(feature = c)[['feature', 'value', 'count']]
        dfs.append(c_counts)
        
        s_c_counts = df.groupby([c, 'sex']).count()[['age']].reset_index().sort_values(by=['sex', 'age'], ascending=False)\
                       .rename({'age':'count', c:'value'}, axis=1).assign(feature=c)[['feature', 'value', 'sex', 'count']]
        dfs_s.append(s_c_counts)
        
    cat_counts = pd.concat(dfs)
    cat_s_counts = pd.concat(dfs_s)
    return cat_counts, cat_s_counts


def statistical_parity_difference(y_true, y_pred=None, *, prot_attr=None,
                                  priv_group=1, pos_label=1, sample_weight=None):
    r"""Difference in selection rates.
    .. math::
        Pr(\hat{Y} = \text{pos_label} | D = \text{unprivileged})
        - Pr(\hat{Y} = \text{pos_label} | D = \text{privileged})
    Note:
        If only y_true is provided, this will return the difference in base
        rates (statistical parity difference of the original dataset). If both
        y_true and y_pred are provided, only y_pred is used.
    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Statistical parity difference.
    See also:
        :func:`selection_rate`, :func:`base_rate`
    """
    rate = base_rate if y_pred is None else selection_rate
    return difference(rate, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)

def difference(func, y_true, y_pred=None, prot_attr=None, priv_group=1,
               sample_weight=None, **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.
    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.
    Unprivileged group is taken to be the inverse of the privileged group.
    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.
    Returns:
        scalar: Difference in metric value for unprivileged and privileged
        groups.
    Examples:
        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> difference(precision_score, y, y_pred, prot_attr='sex',
        ... priv_group='male')
        -0.06955430006277463
    """
    groups, _ = check_groups(y_true, prot_attr)
    idx = (groups == priv_group)
    unpriv = [y[~idx] for y in (y_true, y_pred) if y is not None]
    priv = [y[idx] for y in (y_true, y_pred) if y is not None]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)

def selection_rate(y_true, y_pred, *, pos_label=1, sample_weight=None):
    r"""Compute the selection rate, :math:`Pr(\hat{Y} = \text{pos_label}) =
    \frac{TP + FP}{P + N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values. Ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Selection rate.
    """
    return base_rate(y_pred, pos_label=pos_label, sample_weight=sample_weight)

def base_rate(y_true, y_pred=None, *, pos_label=1, sample_weight=None):
    r"""Compute the base rate, :math:`Pr(Y = \text{pos_label}) = \frac{P}{P+N}`.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
    Returns:
        float: Base rate.
    """
    idx = (y_true == pos_label)
    return np.average(idx, weights=sample_weight)
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
