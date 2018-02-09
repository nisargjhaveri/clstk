from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def evaluate(y_pred, y_test, output=True):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearsonR = pearsonr(y_pred, y_test)
    spearmanR = spearmanr(y_pred, y_test)

    if output:
        print "MSE:", mse
        print "MAE:", mae
        print "Pearson's r:", pearsonR
        print "Spearman r:", tuple(spearmanR)

    return {
        "MSE": mse,
        "MAE": mae,
        "pearsonR": pearsonR,
        "spearmanR": spearmanR
    }
