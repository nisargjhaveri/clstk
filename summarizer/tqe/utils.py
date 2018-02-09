from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def evaluate(y_pred, y_test, output=True):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearsonR = pearsonr(y_pred, y_test)
    spearmanR = spearmanr(y_pred, y_test)

    if output:
        print "\t".join([
            "MSE", "MAE", "PCC", "p-value  ", "SCC", "p-value  "
        ])
        print "\t".join([
            ("%1.5f" % mse),
            ("%1.5f" % mae),
            ("%1.5f" % pearsonR[0]),
            ("%.3e" % pearsonR[1]),
            ("%1.5f" % spearmanR[0]),
            ("%.3e" % spearmanR[1]),
        ])

    return {
        "MSE": mse,
        "MAE": mae,
        "pearsonR": pearsonR,
        "spearmanR": spearmanR
    }
