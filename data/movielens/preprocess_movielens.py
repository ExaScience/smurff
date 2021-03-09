#!/usr/bin/python 

import pandas as pd
import numpy as np

movies = pd.read_csv("merged.csv")

genres = pd.DataFrame([ genre.split("|") for genre in movies["genres_x"] ])
print(f"genres: {genres.shape}")

flat = genres.values.flatten()
flat = np.delete(flat, np.where(flat == None))
unique = np.unique(flat)
print(f"unique genres: {len(np.unique(flat))}")


# 	movieId	title_x	genres_x	imdbId	tmdbId	id	title_y	tagline	description	genres_y	keywords	date	collection	runtime	revenue	budget	director	cast	production_companies	production_countries	popularity	average_vote	num_votes	language	imdb_id	poster_url
# 	movieId	title_x	genres_x	imdbId	tmdbId	id	title_y	tagline	description	genres_y	keywords	date	collection	runtime	revenue	budget	director	cast	production_companies	production_countries	popularity	average_vote	num_votes	language	imdb_id	poster_url
# 
# 
# 0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy	114709	862	862	Toy Story		Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences.	animation, comedy, family	jealousy, toy, boy, friendship, friends, rivalry, boy next door, new toy, toy comes to life	30/10/1995	Toy Story Collection	81	373554033	30000000	John Lasseter	Tom Hanks, Tim Allen, Don Rickles, Jim Varney, Wallace Shawn, John Ratzenberger, Annie Potts, John Morris, Erik von Detten, Laurie Metcalf, R. Lee Ermey, Sarah Freeman, Penn Jillette	Pixar Animation Studios	United States of America	21.946943	7.7	5415	en	tt0114709	/rhIRbceoE9lR4veEXuwCC2wARtG.jpg
