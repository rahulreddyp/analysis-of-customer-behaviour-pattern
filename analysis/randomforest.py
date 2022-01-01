from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_preprocessing.csv_to_dataframe import *
from data_preprocessing.data_cleaning import *


def random_forest_classifier(predict):
    print("in classifier")
    df = read_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'))
    columns = ['Marital_Status', 'Income', 'x0_2n Cycle', 'x0_Basic', 'x0_Graduation', 'x0_Master', 'x0_PhD', 'Age']

    for c in columns:
        df[c] = normalize_column(df[c])
    x = df[columns]
    y = df['ValidCustomer']

    ftrain, ftest, ltrain, ltest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
    rfc.fit(ftrain, ltrain)
    tesPred = rfc.predict(ftest)
    dt_accuracy = accuracy_score(ltest, tesPred) * 100
    print(dt_accuracy)
    print(rfc.predict(np.array(predict).reshape(1, -1)))
    return rfc.predict(np.array(predict).reshape(1, -1))


def random_forest_classifier2(predict):
    print("in classifier")
    df = read_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'))
    columns = ['Marital_Status', 'Income', 'x0_2n Cycle', 'x0_Basic', 'x0_Graduation', 'x0_Master', 'x0_PhD', 'Age']

    for c in columns:
        df[c] = normalize_column(df[c])
    x = df[columns]
    y = df['MostSpntOn']

    ftrain, ftest, ltrain, ltest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
    rfc.fit(ftrain, ltrain)
    tesPred = rfc.predict(ftest)
    dt_accuracy = accuracy_score(ltest, tesPred) * 100
    print(dt_accuracy)
    print(rfc.predict(np.array(predict).reshape(1, -1)))
    return rfc.predict(np.array(predict).reshape(1, -1))


if __name__ == "__main__":
    random_forest_classifier(x=1, y=4)
