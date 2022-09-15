import json
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats

from scipy.spatial.distance import pdist, squareform

from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier


def get_p_string(p):
    if p >= 0.05:
        p_str = "-"
    elif 0.01 <= p < 0.05:
        p_str = "*"
    elif 0.001 <= p < 0.01:
        p_str = "**"
    else:
        p_str = "***"

    return p_str


def stat_print(list_x, name, precision=4, quantile=False):
    if quantile:
        print(
            "Mean " + name + " =", np.round(np.nanmean(list_x), precision),
            "+/-", np.round(np.nanstd(list_x), precision),
            "; Median =", np.round(np.nanmedian(list_x), precision),
            "; Min. = ", np.round(np.nanmin(list_x), precision),
            "; Max. = ", np.round(np.nanmax(list_x), precision),
            "\n Quantile 1%", np.round(np.nanquantile(list_x, 0.01), precision),
            "Quantile 99%", np.round(np.nanquantile(list_x, 0.99), precision)
        )
    else:
        print(
            "Mean " + name + " =", np.round(np.nanmean(list_x), precision),
            "+/-", np.round(np.nanstd(list_x), precision),
            "; Median =", np.round(np.nanmedian(list_x), precision),
            "; Min. = ", np.round(np.nanmin(list_x), precision),
            "; Max. = ", np.round(np.nanmax(list_x), precision)
        )


def obtain_stats(col_x):
    dict_result = {
        "min": np.nanmin( col_x ),
        "perct25": np.nanpercentile( col_x, 25 ),
        "mean": np.nanmean( col_x ),
        "median": np.nanmedian( col_x ),
        "std": np.nanstd( col_x ),
        "perct75": np.nanpercentile( col_x, 75 ),
        "max": np.nanmax( col_x )
    }
    
    return dict_result

        
def make_binary_plot(data, x_feat, x_name, y_feat, y_name, violin_plot=False):
    
    group_list = [df_g[y_feat].tolist() for g, df_g in data.groupby(x_feat)]
    _, pvalue = stats.f_oneway(*group_list)
    
    print("P-value", pvalue, "with significance", get_p_string(pvalue))
    
    plt.figure(figsize=(10, 5))
    
    if violin_plot:
        sns.violinplot(data=data, x=x_feat, y=y_feat)
    else:
        sns.boxplot(data=data, x=x_feat, y=y_feat)
        
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.show()
    

def make_scatter_plot(df, x_feat, x_name, y_feat, y_name, alpha_th=0.6, fontsize_th=14, lr_mode=True):
    data_idx = df[[x_feat, y_feat]].dropna().index.values

    corr_v, corr_pvalue = stats.pearsonr(df.loc[data_idx, x_feat].values, df.loc[data_idx, y_feat].values)
    print("Correlation value", corr_v)
    print("P-value", get_p_string(corr_pvalue))

    plt.figure(figsize=(12, 5))

    if lr_mode:
        lr = linear_model.LinearRegression(n_jobs=-1)
        lr.fit(df.loc[data_idx, x_feat].values.reshape(-1, 1), df.loc[data_idx, y_feat])
        y_pred = lr.predict(df.loc[data_idx, x_feat].values.reshape(-1, 1))

        print("Coefficients: \n", lr.coef_, "\nIntercept: \n", lr.intercept_)
        print("R-square: %.2f" % r2_score(df.loc[data_idx, y_feat], y_pred))

        plt.plot(df.loc[data_idx, x_feat], y_pred, c="red", label="Linear Regression")
        plt.legend(fontsize=12)
        delta_int = np.abs(np.nanmax(df[y_feat]) - np.nanmin(df[y_feat]))
        plt.ylim(np.nanmin(df[y_feat]) - 0.1 * delta_int, np.nanmax(df[y_feat]) + 0.1 * delta_int)

    plt.scatter(df[x_feat], df[y_feat], alpha=0.6)

    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)

    plt.show()

    
def get_feature_importance(data, target,
                           features_to_use,
                           num_repeats=4, num_splits=10, 
                           model=ExtraTreesClassifier(n_estimators=400, max_depth=30, random_state=0), 
                           verbose=False):
    
    rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=rdm_th)

    y_data = data[target]
    x_data = data[features_to_use]

    save_feature_imp = []
    metrcis_dict = {
        "Accuracy": [],
        "Kappa": [],
        "RocAUC": [],
    }
    for i_split, (train_index, test_index) in enumerate(rskf.split(x_data, y_data)):
        
        if verbose:
            print("Split", i_split+1, end="\r")
        
        x_train, x_test = x_data.loc[train_index], x_data.loc[test_index]
        y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]

        model.fit(x_train, y_train)

        df_imp_feat = pd.DataFrame(model.feature_importances_).T
        df_imp_feat.columns = features_to_use

        save_feature_imp.append(df_imp_feat)
        
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        
        metrcis_dict["Accuracy"].append(sk_metrics.accuracy_score(y_test, y_pred))
        metrcis_dict["Kappa"].append(sk_metrics.cohen_kappa_score(y_test, y_pred))
        metrcis_dict["RocAUC"].append(sk_metrics.roc_auc_score(y_test, y_pred_proba))
        
    return pd.concat(save_feature_imp), metrcis_dict
    
    
def get_mahalanobis_dist(x_arr, centers=None):
    if centers is None:
        centers = np.mean(x_arr, axis=0)
    cov = np.cov(x_arr.astype(float).T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_arr - centers, inv_covmat)
    mahal = np.dot(left_term, (x_arr - centers).T)

    return mahal.diagonal()