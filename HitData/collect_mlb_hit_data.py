from bs4 import BeautifulSoup
import requests
import pandas as pd

with open("./mlb_hit_data.html") as f:
    soup = BeautifulSoup(f, "html.parser")

data = soup.find_all("tr")

all_data = []
for result in data:
    text_data = result.find_all("span")
    data_list = []
    for result_data in text_data:
        data_list.append(result_data.text)

    all_data.append(data_list)

df = pd.DataFrame(all_data)
df = df.drop(df.columns[[5]], axis=1)
df.columns = ["PlayerName", "Velocity", "Distance", "Angle", "HitType", "Pitcher", "PitchSpeed", "Date"]
df.to_csv("mlb_hit_data.csv")
