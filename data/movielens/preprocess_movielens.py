#!/usr/bin/python 

from pprint import pprint
from numpy.core.fromnumeric import nonzero
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import smurff

movies = pd.read_csv("merged.csv")
movies["movieIdx"] = movies.index # used in sparse matrix

ratings = pd.read_csv("ratings.csv")

ratings["userIdx"] = ratings["userId"] - 1
ratings = pd.merge(ratings, movies, on = "movieId", how = "inner")
ratings_matrix = sp.coo_matrix((ratings['rating'].values,
        (ratings['movieIdx'].values, ratings['userIdx'].values))).tocsr()

movies["genres_x"] = movies["genres_x"].str.replace("|", ",", regex=False)
movies["timestamp"] = pd.to_datetime(movies["date"]).fillna(pd.Timestamp(0.)).astype('int64') // 10**9

lang_ids = { l : i for i,l in enumerate(movies["language"].unique()) }
movies["language_id"] = movies["language"].map(lang_ids)

numerical_columns = [
    "timestamp",
    "runtime",
    "revenue",
    "budget",
    "popularity",
    "average_vote",
    "num_votes",
    "language_id",
]

category_columns = [
    "genres_x",
    "genres_y",
    #"keywords",
    #"director",
    #"cast",
    #"production_companies",
    "production_countries",
]

movies[numerical_columns] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(movies[numerical_columns])
movies[numerical_columns] = StandardScaler().fit_transform(movies[numerical_columns])
movies.to_csv("movies_normalized.csv")

### ---  category

movies[category_columns] = movies[category_columns].fillna("")

for col in category_columns:
    movies[col] = movies[col].str.split(", ?")

features_vectorizer = DictVectorizer()
categories_dict = movies[category_columns].to_dict(orient = 'records')
categories_matrix = features_vectorizer.fit_transform(categories_dict)

for col,name in enumerate(features_vectorizer.get_feature_names()):
    data = categories_matrix[:,col].data
    num_nan = np.isnan(data).sum()
    if num_nan > 0:
        print(f"{name}: {num_nan} / {len(data)} - {data}")


features_matrix = sp.hstack((categories_matrix, sp.csc_matrix(movies[numerical_columns])))

import pickle

pickle.dump(features_vectorizer, open("movies_features_vectorizer.pickle", "wb"))
pickle.dump(features_matrix,     open("movie_features_matrix.pickle", "wb"))
pickle.dump(ratings_matrix,      open("ratings_matrix.pickle", "wb"))

smurff.matrix_io.write_matrix("ratings.sdm", ratings_matrix)
smurff.matrix_io.write_matrix("features.sdm", features_matrix)

#--- only save 1000 movies and 1000 users
ratings_matrix = ratings_matrix.tocsr()
features_matrix = features_matrix.tocsr()

nnz_per_movie = [ ratings_matrix[r,:].nnz for r in range(ratings_matrix.shape[0]) ]
popular_movies = np.argsort(nnz_per_movie)
ratings_matrix_1k = ratings_matrix[popular_movies[-1000:], :]
features_matrix_1k = features_matrix[popular_movies[-1000:], :]

ratings_matrix_1k = ratings_matrix_1k.tocsc()
nnz_per_user = [ ratings_matrix_1k[:,r].nnz for r in range(ratings_matrix_1k.shape[1]) ]
popular_users = np.argsort(nnz_per_user)
ratings_matrix_1k = ratings_matrix_1k[:, popular_users[-1000:]]

smurff.matrix_io.write_matrix("ratings_1k.sdm", ratings_matrix_1k)
smurff.matrix_io.write_matrix("features_1k.sdm", features_matrix_1k)