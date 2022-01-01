from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from data_preprocessing.data_cleaning import *
from data_preprocessing.data_insights import *
from data_preprocessing.data_encoding import *
from data_preprocessing.csv_to_dataframe import *
from data_preprocessing.dataframe_to_csv import *


def process_marketing_campaign():
    df = read_dataset(Path('..', 'datasets', 'raw', 'marketing_campaign.csv'), '\t')
    df['Marital_Status'].replace(
        {'Single': 0, 'Together': 1, 'Married': 1, 'Divorced': 0, 'Widow': 0, 'Alone': 0, 'Absurd': 0, 'YOLO': 0},
        inplace=True)
    df['maritalstatus'] = df['Marital_Status'].replace(
        {'Single': 'unmarried', 'Together': 'unmarried', 'Married': 'married', 'Divorced': 'unmarried',
         'Widow': 'unmarried', 'Alone': 'unmarried', 'Absurd': 'unmarried', 'YOLO': 'unmarried'}
    )
    df['numberofkids'] = df['Kidhome'] + df['Teenhome']
    df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)
    tmpSeries = pd.DatetimeIndex(df['Dt_Customer']).year
    df['Age'] = tmpSeries - df['Year_Birth']
    df['ValidCustomer'] = df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df[
        'MntGoldProds'] + df['MntWines']
    df['amountspent'] = df['ValidCustomer']
    mtspentmean = get_column_mean(df, 'ValidCustomer')
    # df['Age']= df['Age'].apply(lambda x: 'Teen' if x<20 else 'Adult' if x>=20 and x<50 else 'Elder')

    df['ValidCustomer'] = df['ValidCustomer'].apply(lambda x: 'yes' if x > mtspentmean else 'no')
    categorical_columns = get_text_categorical_columns(df)
    categorical_columns.remove('ValidCustomer')
    binary_columns = get_binary_columns(df)
    numeric_columns = get_numeric_columns(df)
    categorical_columns.remove('Dt_Customer')

    for bc in binary_columns:
        numeric_columns.remove(bc)
        df = fix_nans(df, bc)
        df = fix_outliers(df, bc)
    for cc in categorical_columns:
        df = fix_nans(df, cc)
        df = fix_outliers(df, cc)
        ohe = generate_one_hot_encoder(df[cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))
    for nc in numeric_columns:
        df = fix_nans(df, nc, int(get_column_mean(df, nc)))
        df = fix_outliers(df, nc)
        df = fix_numeric_wrong_values(df, nc, WrongValueNumericRule.MUST_BE_POSITIVE)
        df = fix_nans(df, nc, get_column_mean(df, nc))
        # df[nc] = normalize_column(df[nc])
        write_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'), df, encoding='utf-8', sep=',')

    print(df)


def customeranalysis():
    df = read_dataset(Path('..', 'datasets', 'raw', 'marketing_campaign.csv'), '\t')
    df['Marital_Status'].replace(
        {'Single': 'unmarried', 'Together': 'unmarried', 'Married': 'married', 'Divorced': 'unmarried',
         'Widow': 'unmarried', 'Alone': 'unmarried', 'Absurd': 'unmarried', 'YOLO': 'unmarried'},
        inplace=True)
    df['numberofkids'] = df['Kidhome'] + df['Teenhome']
    df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)
    tmpSeries = pd.DatetimeIndex(df['Dt_Customer']).year
    df['Age'] = tmpSeries - df['Year_Birth']
    df['amountspent'] = df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df[
        'MntGoldProds'] + df['MntWines']
    df['Age'] = df['Age'].apply(lambda x: 'Teen' if x < 20 else 'Adult' if x >= 20 and x < 50 else 'Elder')
    df['numofvisits'] = df['NumStorePurchases'] + df['NumWebVisitsMonth']
    df['totalpurchaces'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases']
    print(get_column_number_of_duplicates(df, 'ID'))
    df.drop(['MntWines', 'MntFruits', 'MntMeatProducts',
             'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
             'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
             'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
             'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
             'Complain', 'Z_CostContact', 'Z_Revenue'], inplace=True, axis=1)
    numeric_columns = get_numeric_columns(df)
    for nc in numeric_columns:
        df = fix_nans(df, nc, int(get_column_mean(df, nc)))
        df = fix_outliers(df, nc)
        df = fix_numeric_wrong_values(df, nc, WrongValueNumericRule.MUST_BE_POSITIVE)

    write_dataset(Path('..', 'datasets', 'processed', 'cusanalysis.csv'), df, encoding='utf-8', sep=',')
    print(df)
    pass


def customeragedu():
    df = read_dataset(Path('..', 'datasets', 'processed', 'cusanalysis.csv'))

    degrees = df['Education'].unique()
    dflist, education, percentage = [], [], []
    for degree in degrees:
        tmpdf = df[df['Education'] == degree].copy()
        tmp = tmpdf.groupby(['Age'], as_index=False, sort=False)['Education'].count()
        dflist.append(tmp)
        for i in range(len(tmp)):
            percentage.append((tmp.loc[i, 'Education'] / tmp['Education'].sum()) * 100)
            education.append(degree)
    finaldf = pd.concat(dflist)
    finaldf['Education'] = education
    finaldf['Percentage'] = percentage
    write_dataset(Path('..', 'datasets', 'processed', 'customerhirarchy.csv'), finaldf, encoding='utf-8', sep=',')


def productinsight():
    df = read_dataset(Path('..', 'datasets', 'raw', 'marketing_campaign.csv'), '\t')
    df = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
    for col in df.columns:
        df[col].fillna(0, inplace=True)
    write_dataset(Path('..', 'datasets', 'processed', 'product.csv'), df, encoding='utf-8', sep=',')
    print(df)


def cpanalysis():
    df = read_dataset(Path('..', 'datasets', 'raw', 'marketing_campaign.csv'), '\t')
    df['Marital_Status'].replace(
        {'Single': 'unmarried', 'Together': 'married', 'Married': 'married', 'Divorced': 'divorced',
         'Widow': 'unmarried', 'Alone': 'unmarried', 'Absurd': 'unmarried', 'YOLO': 'unmarried'},
        inplace=True)
    df['numberofkids'] = df['Kidhome'] + df['Teenhome']
    df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)
    tmpSeries = pd.DatetimeIndex(df['Dt_Customer']).year
    df['Age'] = tmpSeries - df['Year_Birth']
    df['AgeCategory'] = df['Age'].apply(lambda x: 'Teen' if x < 20 else 'Adult' if x >= 20 and x < 50 else 'Elder')
    write_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'), df, encoding='utf-8', sep=',')


if __name__ == "__main__":
    customeranalysis()
