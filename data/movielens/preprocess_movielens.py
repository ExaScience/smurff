#!/usr/bin/python 

from pprint import pprint
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer

movies = pd.read_csv("merged.csv")
movies["movieIdx"] = movies.index # used in sparse matrix

ratings = pd.read_csv("ratings.csv")

ratings["userIdx"] = ratings["userId"] - 1
ratings = pd.merge(ratings, movies, on = "movieId", how = "inner")
ratings_matrix = sp.coo_matrix((ratings['rating'].values,
        (ratings['movieIdx'].values, ratings['userIdx'].values)))


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
features_vectorizer = DictVectorizer()
features_matrix = features_vectorizer.fit_transform(selected)

import pickle

pickle.dump(features_vectorizer, open("movies_features_vectorizer.pickle", "wb"))
pickle.dump(features_matrix,     open("movie_features_matrix.pickle", "wb"))
pickle.dump(ratings_matrix,      open("ratings_matrix.pickle", "wb"))