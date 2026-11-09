import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_coffee_data():

    df_coffee = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")
    
    return df_coffee

def train_model_1(df_coffee):

    df_train, df_test = train_test_split(df_coffee, test_size=.2)
    # df_test.to_csv('./data/model_1_holdout_set.csv')

    X = df_train[['100g_USD']]
    y = df_train['rating']

    lm_0 = LinearRegression()
    lm_0.fit(X, y)

    return lm_0

def roast_category(df):
    df['roast_cat'] = df['roast'].astype('category').cat.codes
    return None

def train_model_2(df_coffee):
    roast_category(df_coffee)

    df_train, df_test = train_test_split(df_coffee, test_size=.2)

    X = df_train[['100g_USD', 'roast_cat']]
    y = df_train['rating']

    lm_0 = DecisionTreeRegressor()
    lm_0.fit(X, y)

    return lm_0


if __name__ == '__main__':
    df_coffee = get_coffee_data()

    model_1 = train_model_1(df_coffee)

    with open('./model_1.pickle', 'wb') as f:
        pickle.dump(model_1, f)

    model_2 = train_model_2(df_coffee)
    
    with open('./model_2.pickle', 'wb') as f:
        pickle.dump(model_2, f)
