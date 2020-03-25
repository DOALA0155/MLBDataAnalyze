import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mlb_homerun_analyze_data.csv")
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
    df_y = pd.DataFrame(labels)
    df_y.columns = ["Cluster"]
    df_pairplot = pd.concat([df, df_y], axis=1)

    pg = sns.pairplot(df_pairplot, hue="Cluster")
    pg.savefig("./mlb_homerun_clustering.png")

    cluster_average = []
    for i in range(2):
        cluster = data[labels==i]
        df = pd.DataFrame(cluster)
        df.columns = ["Distance", "Velocity", "Angle", "Height", "PitchSpeed"]
        df_mean = df.mean()
        print("Cluster {}".format(i))
        print(df_mean)

clustering(df)
