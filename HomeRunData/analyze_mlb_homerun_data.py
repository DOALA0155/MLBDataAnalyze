import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mlb_homerun_data.csv")
df = df.drop(df.columns[[0]], axis=1)

def regression(df):
    angle = df["Angle"].values.reshape(-1, 1)
    height = df["Height"].values.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(angle, height)
    score = lr.score(angle, height)
    pred_y = lr.predict(np.linspace(20, 40, 100).reshape(-1, 1))

    plt.plot(np.linspace(20, 40, 100), pred_y)
    plt.scatter(angle, height)
    plt.xlabel("Angle")
    plt.ylabel("Height")
    plt.show()

    return score

def clustering(df):
    data = df.values
    sd = StandardScaler()
    data = sd.fit_transform(data)

    db = KMeans(n_clusters=2)
    db.fit(data)

    labels = db.labels_
    cluster0 = data[labels==0]
    cluster1 = data[labels==1]

    clusters = [cluster0, cluster1]
    colors = ["b", "g", "r"]
    cluster_ids = [[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    cluster_name = {0: "Distance", 1: "Velocity", 2: "Angle", 3: "Height", 4: "PitchSpeed"}

    for cluster_id in cluster_ids:
        for cluster, c in zip(clusters, colors):
            plt.scatter(cluster[:, cluster_id[0]], cluster[:, cluster_id[1]], color=c)
            plt.xlabel(cluster_name[cluster_id[0]])
            plt.ylabel(cluster_name[cluster_id[1]])
        plt.show()

clustering(df)
