import math
from typing import Tuple
# from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from data_preprocessing.process_data import *
from data_preprocessing.csv_to_dataframe import *
from plotly.subplots import make_subplots


def pie(df, values, names):
    fig = px.pie(df, names=values, values=names)
    return fig


def bar(dataframe, x, y):
    dataframe[y] = dataframe[y]
    fig = px.bar(dataframe, x=x, y=y)
    return fig


def line(dataframe, x, y):
    print(x)
    print(y)
    fig = px.line(dataframe, x=x, y=y)
    return fig


def compositelinebar(dataframe, x, y):
    fig = go.Figure()
    x = list(dataframe[x])
    y = list(dataframe[y])
    fig.add_trace(
        go.Scatter(x=x, y=y
                   ))

    fig.add_trace(
        go.Bar(x=x, y=y

               ))
    return fig


def iciecle(dataframe):
    fig = px.icicle(dataframe, path=[px.Constant("Age category"), 'Education', 'Age'], values='Percentage')
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


def sunburst(dataframe):
    fig = px.sunburst(dataframe, path=['Education', 'Age'], values='Percentage')
    return fig


def horizontalbar(dataframe):
    # dataframe = read_dataset(Path('..', 'datasets', 'processed', 'product.csv'))
    dic = {}
    for column in dataframe.columns:
        dic[column] = 0
        for record in dataframe[column]:
            if record != 0:
                dic[column] = dic[column] + 1
    tmpdic = {}
    tmpdic['products'] = list(dic.keys())
    tmpdic['numoftimebought'] = list(dic.values())
    dataframe = pd.DataFrame(tmpdic)
    fig = px.bar(dataframe, x="numoftimebought", y="products", orientation='h')
    return fig


def heatmap(dataframe):
    correlation = dataframe.corr()
    fig = px.imshow(correlation, color_continuous_scale='sunsetdark')
    return fig


def subplotss(dataframe, x, z):
    products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    plots = []
    rows = 2
    cols = 3
    titles = ['Wine and ' + x + " bar plot", 'Fruits and ' + x + " bar plot", 'Meat and ' + x + " bar plot",
              'Fish and ' + x + " bar plot",
              'Sweet and ' + x + " Bar plot", 'Gold and ' + x + " bar plot"]
    if z == 'mean':
        for product in products:
            print(product)
            tmpdf = dataframe.groupby(x, as_index=False, sort=False)[product].mean()
            plots.append(px.bar(tmpdf, x=x, y=product))
    if z == 'max':
        for product in products:
            print(product)
            tmpdf = dataframe.groupby(x, as_index=False, sort=False)[product].max()
            plots.append(px.bar(tmpdf, x=x, y=product))
    if z == 'min':
        for product in products:
            print(product)
            tmpdf = dataframe.groupby(x, as_index=False, sort=False)[product].min()
            plots.append(px.bar(tmpdf, x=x, y=product))

    fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
    index = 0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.add_trace(
                plots[index].data[0], row=i, col=j
            )
            index = index + 1
    return fig


def tabel(df, fd, bv):
    tmp = df.groupby([fd], as_index=False, sort=False)[
        'Response', 'amountspent', 'numofvisits', 'totalpurchaces'].mean()
    tmp = tmp[tmp[fd] == bv]
    print(tmp)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(tmp.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[tmp[fd], tmp['Response'], tmp['amountspent'], tmp['numofvisits'], tmp['totalpurchaces']],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig
    # fig.show()


def groupedbar(cat):
    # cat='Marital_Status'
    print(cat)
    df = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
    marketing = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    tmpdf = df.groupby([cat], as_index=False, sort=False)[marketing].sum()
    bars = []
    xvals = df[cat].unique()

    for m in marketing:
        bars.append(go.Bar(name=m, x=xvals, y=list(tmpdf[m])))

    fig = go.Figure(data=bars)
    fig.update_layout(barmode='group')
    # fig.show()
    return fig


def barmarketing():
    df = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
    marketing = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    tmp = {}
    for m in marketing:
        tmp[m] = (df[m] == 0).sum(axis=0)
    fig = px.bar(x=tmp.keys(), y=tmp.values())
    return fig


def scatter(v1, v2):
    cat = v1
    df = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
    pc = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    tmpdf = df.groupby([cat], as_index=False, sort=False)[pc].sum()

    fig = px.line(tmpdf, x=cat, y=v2)

    return fig


def piemarketing():
    dataframe = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
    pc = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    tmp = {}
    for tp in pc:
        tmp[tp] = (dataframe[tp] == 0).sum(axis=0)
    fig = px.pie(names=tmp.keys(), values=tmp.values())
    return fig


def funnel():
    df = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
    pc = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['totalpurchases'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df[
        'NumStorePurchases']
    names = []
    totalpurchases = []
    names.append('totalpurchases')
    totalpurchases.append(df['totalpurchases'].sum())
    names.append('NumStorePurchases')
    totalpurchases.append(df['NumStorePurchases'].sum())
    names.append('NumWebPurchases')
    totalpurchases.append(df['NumWebPurchases'].sum())
    names.append('NumCatalogPurchases')
    totalpurchases.append(df['NumCatalogPurchases'].sum())
    names.append('NumDealsPurchases')
    totalpurchases.append(df['NumDealsPurchases'].sum())
    data = dict(
        number=totalpurchases,
        stage=names)
    fig = px.funnel(data, x='number', y='stage')
    # fig.show()
    return fig


def potentialbar(type):
    df = read_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'))
    columns = ['Marital_Status', 'Income', 'x0_2n Cycle', 'x0_Basic', 'x0_Graduation', 'x0_Master', 'x0_PhD', 'Age']
    # type='MostSpntOn'
    # tt='ValidCustomer'
    for c in columns:
        df[c] = normalize_column(df[c])
    x = df[columns]
    y = df[type]

    ftrain, ftest, ltrain, ltest = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
    rfc.fit(ftrain, ltrain)
    testPred = rfc.predict(ftest)
    accuracy = accuracy_score(ltest, testPred) * 100
    precision = precision_score(ltest, testPred, average='micro') * 100
    recall = recall_score(ltest, testPred, average='micro') * 100
    accdict = dict(accuracy=accuracy, precision=precision,
                   recall=recall)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(accdict.keys()),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=list(accdict.values()),
                   fill_color='lavender',
                   align='left'))])
    return fig


def profiling():
    df = read_dataset(Path('..', 'datasets', 'raw', 'marketing_campaign.csv'), '\t')
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title="Pandas Profiling Report")
    if 'profiling.html' is not None:
        profile.to_file('profiling.html')
    return profile.to_json()


def svmMetricts():
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
    svmR_mse = mean_squared_error(ltest, yPred)
    svmR_mae = mean_absolute_error(ltest, yPred)
    error = dict(Meansquarederror=svmR_mse, Mabsoluteerror=svmR_mae)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(error.keys()),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=list(error.values()),
                   fill_color='lavender',
                   align='left'))])
    return fig


if __name__ == "__main__":
    df = read_dataset(Path('..', 'datasets', 'processed', 'customerhirarchy.csv'))
    svmMetricts()
