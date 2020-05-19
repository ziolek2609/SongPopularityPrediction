import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


# wczytanie danych
raw_data = pd.read_csv("/Users/ziole/Desktop/pro/song_data.csv")


# przekszatałcenie danych
data = raw_data.copy()
songs=data.drop_duplicates(keep="first")
songs_data = songs.drop('song_name', axis=1)
minmax = MinMaxScaler(feature_range=(0,1))
minmax_df = minmax.fit_transform(songs_data)
songs_data_scaled = pd.DataFrame(minmax_df, columns=songs_data.columns)


# podział na zbiór testowy i treningowy
train_dataset = songs_data_scaled.sample(frac=0.8, random_state=0)
test_dataset = songs_data_scaled.drop(train_dataset.index)
train_labels = train_dataset.pop('song_popularity')
test_labels = test_dataset.pop('song_popularity')

train_dataset.to_csv("/Users/ziole/Desktop/pro/train_dataset.csv")
test_dataset.to_csv("/Users/ziole/Desktop/pro/test_dataset.csv")

train_labels.to_csv("/Users/ziole/Desktop/pro/train_labels.csv")
test_labels.to_csv("/Users/ziole/Desktop/pro/test_labels.csv")
