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

