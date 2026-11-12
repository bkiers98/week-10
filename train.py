import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def get_coffee_data():
    """Reads in coffee_analysis.csv as a pandas DataFrame"""
    df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    return df_coffee

def train_model_1(df_coffee):
    """Trains a linear regression model on df_coffee"""
    X = df_coffee[['100g_USD']]
    y = df_coffee['rating']

    lm = LinearRegression()
    lm.fit(X, y)

    return lm

def roast_category(roast):
    """Assigns numerical category to incoming roast strings"""
    roast_codes = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4,
    }

    try:
        return roast_codes[roast]
    except KeyError:
        return np.nan

def train_model_2(df_coffee):
    """Trains a decision tree regressor model on df_coffee"""
    df_coffee = df_coffee.copy()
    df_coffee['roast_cat'] = df_coffee['roast'].map(roast_category)

    X = df_coffee[['100g_USD', 'roast_cat']]
    y = df_coffee['rating']

    dtr = DecisionTreeRegressor()
    dtr.fit(X, y)

    return dtr

if __name__ == '__main__':
    df_coffee = get_coffee_data()

    model_1 = train_model_1(df_coffee)

    with open('./model_1.pickle', 'wb') as f:
        pickle.dump(model_1, f)

    model_2 = train_model_2(df_coffee)

    with open('./model_2.pickle', 'wb') as f:
        pickle.dump(model_2, f)
