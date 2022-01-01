from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from data_preprocessing.csv_to_dataframe import read_dataset
from data_preprocessing.data_cleaning import normalize_column
from data_preprocessing.data_insights import *


def svm(input):
    regressor = SVR(kernel='rbf')
    df = read_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'))
    columns = ['Marital_Status', 'Income', 'x0_2n Cycle', 'x0_Basic', 'x0_Graduation', 'x0_Master', 'x0_PhD', 'Age']
    tmpdf = df.copy()
    for c in columns:
        df[c] = normalize_column(df[c])

    df['amountspent'] = normalize_column(df['amountspent'])
    x = df[columns]
    y = df['amountspent']
    ftrain, ftest, ltrain, ltest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    regressor.fit(ftrain, ltrain)
    yPred = regressor.predict(ftest)
    dt_accuracy = mean_squared_error(ltest, yPred)
    imin = get_column_min(tmpdf, 'Income')
    imax = get_column_max(tmpdf, 'Income')
    amin = get_column_min(tmpdf, 'Age')
    amax = get_column_max(tmpdf, 'Age')
    asmin = get_column_min(tmpdf, 'amountspent')
    asmax = get_column_max(tmpdf, 'amountspent')
    pv = regressor.predict(
        np.array([0, (33812 - imin) / (imax - imin), 1, 0, 0, 0, 0, (27 - amin) / (amax - amin)]).reshape(1, -1))
    pv = regressor.predict(np.array(input).reshape(1, -1))
    print((pv[0] * (asmax - asmin)) + asmin)
    return (pv[0] * (asmax - asmin)) + asmin


if __name__ == "__main__":
    svm(x=4, y=3)
