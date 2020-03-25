import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

def preprocessing_data(df):
    df_x = df.drop(df.columns[[3]], axis=1).values

    df_y = df.loc[:, "HitType"]
    le = LabelEncoder()
    df_y_ohe = le.fit_transform(df_y.values)

    return df_x, df_y_ohe

def pairplot(df):
    pg = sns.pairplot(df, hue="HitType")
    pg.savefig("./mlb_hit_data.png")

def scaling(x):
    sd = StandardScaler()
    df_scaled = sd.fit_transform(x)
    return df_scaled

def predicted_pairplot(df, model):
    x, y = preprocessing_data(df)
    x_scaled = scaling(x)
    column_list = ["Velovity", "Distance", "Angle", "PitchSpeed"]

    y_pred = model.predict(x_scaled)

    df_x_pred = pd.DataFrame(x)
    df_x_pred.columns = column_list

    df_y = pd.DataFrame(y_pred)
    df_y.columns = ["HitType"]

    df = pd.concat([df_x_pred, df_y], axis=1)

    pg = sns.pairplot(df, hue="HitType")
    pg.savefig("./mlb_hit_svm_predict.png")

def train_model(df, model):
    x, y = preprocessing_data(df)
    x_scaled = scaling(x)

    model.fit(x_scaled, y)
    score = model.score(x_scaled, y)
    print(score)
    return model

df = pd.read_csv("mlb_hit_data.csv")
df = df.drop(df.columns[[0, 1, 6, 8]], axis=1)

svm = SVC()
svm = train_model(df, svm)
predicted_pairplot(df, svm)
