import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn.metrics as sk_metrics 


def get_model_metrics(X_data, Y_data, model, n_splits=10, n_repeats=4, random_state=0):
    
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )

    metrics_report_list = []
    for i_rskf, (train_idx, test_idx) in enumerate(rskf.split(X_data, Y_data)):

        print("Iteration", i_rskf, end="\r")

        x_train, x_test = X_data.loc[train_idx], X_data.loc[test_idx]
        y_train, y_test = Y_data.loc[train_idx], Y_data.loc[test_idx]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc_score = sk_metrics.accuracy_score(y_test, y_pred)
        kappa_score = sk_metrics.cohen_kappa_score(y_test, y_pred)
        auc_score = sk_metrics.roc_auc_score(y_test, y_pred)

        metrics_report_list.append({
            "Iteration": i_rskf,
            "Accuracy": acc_score,
            "CohenKappa": kappa_score,
            "AUC": auc_score
        })

    return pd.DataFrame(metrics_report_list)