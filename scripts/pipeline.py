import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def load_data(data_url):
    """Carrega os dados através da URL passada."""
    return pd.read_csv(data_url, encoding='latin-1')


def remove_outliers(data, column):
    """Remove os valores que estão fora da amplitude
       interquartil (IQR), ou seja, os outliers."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3-Q1
    limite_inf = Q1-1.5*IQR
    limite_sup = Q3+1.5*IQR
    data = data.loc[(data[column] > limite_inf) & (data[column] < limite_sup)]
    return data


def clean_data(df):
    """Prepara e limpa os dados."""
    df = df_raw.copy()
    df.drop('PctSomeCol18_24', axis=1, inplace=True)
    df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].mean(), inplace=True)
    df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].mean(), inplace=True)
    df.drop('binnedInc', axis=1, inplace=True)
    df.drop(['MedianAgeMale', 'MedianAgeFemale'], axis=1, inplace=True)
    df.drop('Geography', axis=1, inplace=True)
    variaveis_muito_correlacionadas = ['avgAnnCount', 'popEst2015', 'povertyPercent',
                                       'PctPublicCoverage', 'PctPublicCoverageAlone',
                                       'PctPrivateCoverage', 'PctPrivateCoverageAlone',
                                       'PctEmpPrivCoverage', 'PercentMarried']
    df.drop(variaveis_muito_correlacionadas, axis=1, inplace=True)
    colunas_a_limpar = ['TARGET_deathRate', 'avgDeathsPerYear', 'MedianAge',
                        'medIncome', 'AvgHouseholdSize']

    for coluna in colunas_a_limpar:
        df = remover_outliers(df, coluna)
    
    return df


def split_scale_data(df):
    """Separa os dados e realiza o escalonamento."""
    X = df.drop('TARGET_deathRate', axis=1)
    y = df['TARGET_deathRate']

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lasso_model(X_train, y_train, X_test):
    """Treina o modelo de regressão Lasso."""
    lasso_reg = Lasso(alpha=1.0)
    lasso_reg.fit(X_train, y_train)
    y_test_preds_lasso_reg = lasso_reg.predict(X_test)
    return y_test_preds_lasso_reg


def calculate_metrics(y_real, y_prediction):
    """Calcula as métricas de regressão R2, MAE, MSE e RMSE."""
    r2 = round(lasso_reg.score(X_test, y_test), 3)
    mae = round(metrics.mean_absolute_error(y_test, y_test_preds_lasso_reg), 3)
    mse = round(metrics.mean_squared_error(y_test, y_test_preds_lasso_reg), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_test_preds_lasso_reg)), 3)
    return r2, mae, mse, rmse


def pipeline(data_url):
    """Realiza o passo-a-passo completo do projeto."""
    df_raw = load_data(data_url)
    df_clean = clean_data(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_scale_data(df_clean)
    y_preds = train_lasso_model(X_train, y_train, X_test)
    r2, mae, mse, rmse = calculate_metrics(y_test, y_preds)
    print(f'Regressão Lasso\nR2 = {r2}\nMAE = {mae}\nMSE = {mse}\nRMSE = {rmse}')

if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/diascarolina/predicting-cancer-rates/main/data/cancer_reg.csv'
    pipeline(data_url)