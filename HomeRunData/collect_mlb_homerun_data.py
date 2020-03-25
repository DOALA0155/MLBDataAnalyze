from bs4 import BeautifulSoup
import requests
import pandas as pd

with open("./mlb_homerun_data.html") as f:
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
df.to_csv("mlb_homerun_data.csv")
