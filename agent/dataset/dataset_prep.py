import pandas as pd

# df = pd.read_csv("agent/dataset/movie/Reviews.csv")
# df = df.drop(columns=['scoreSentiment', 'reviewState', 'originalScore'])
# df.to_csv("agent/dataset/movie/Reviews.csv", index=False)

movies = pd.read_csv("agent/dataset/movie/Movies.csv")
reviews = pd.read_csv("agent/dataset/movie/Reviews.csv")
reviews = reviews[reviews['idx']=='ant_man_and_the_wasp_quantumania']
print(len(reviews))