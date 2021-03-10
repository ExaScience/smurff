#!/usr/bin/python 

from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

movies = pd.read_csv("merged.csv")
movies["genres_x"] = movies["genres_x"].str.replace("|", ",", regex=False)
movies["timestamp"] = pd.to_datetime(movies["date"]).fillna(pd.Timestamp(0.)).astype('int64') // 10**9

numerical_columns = [
    "timestamp",
    "runtime",
    "revenue",
    "budget",
    "popularity",
    "average_vote",
    "num_votes",
]

category_columns = [
    "genres_x",
    "genres_y",
    #"keywords",
    #"director",
    #"cast",
    #"production_companies",
    "production_countries",
    "language",
]

for col in category_columns:
    movies[col] = movies[col].str.split(", ?")

selected = movies[numerical_columns + category_columns].to_dict(orient = 'records')
vec = DictVectorizer()
features = vec.fit_transform(selected)