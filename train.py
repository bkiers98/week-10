import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

    y_pred = lm_0.predict(X)
    return y_pred


if __name__ == '__main__':
    df_coffee = get_coffee_data()

    model_1 = train_model_1(df_coffee)

    with open('./model_1.pickle', 'wb') as f:
        pickle.dump(model_1, f)

