#!/usr/bin/python 


import pandas as pd

sideinfo = pd.read_csv("movies_with_sideinfo/movies.csv")
sideinfo["imdbId"] = sideinfo["imdb_id"].str.lstrip(to_strip="tt").fillna(-1).astype(int)

movies   = pd.read_csv("ml-25m/movies.csv").sort_values(by="movieId")
links    = pd.read_csv("ml-25m/links.csv")

movies   = movies.merge(links, on="movieId")
movies   = movies.merge(sideinfo, on="imdbId")

ratings  = pd.read_csv("ml-25m/ratings.csv").sort_values(by="movieId")
    
print(movies.columns)

movies.to_csv("merged.csv")
