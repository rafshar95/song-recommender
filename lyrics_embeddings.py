import numpy as np
import nltk
import pandas as pd
import time
import datetime


def load_lyrics():
  print('Loading lyrics...')
  df =pd.read_csv('data/spotify_songs_with_lyrics.csv')
  print(df)
  df1 = pd.read_csv('data/spotify_summaries__0__0.30.csv')
  df2 = pd.read_csv('data/spotify_summaries__1__0.30.csv')
  df3 = pd.read_csv('data/spotify_summaries__2__0.30.csv')
  df4 = pd.read_csv('data/spotify_summaries__3__0.30.csv')
  df = pd.concat([df1, df2, df3, df4], ignore_index = True)
  print(df)
  df['lyrics'] = df['lyrics'].apply(str)
  return df


def load_embeddings():
  print('Loading word embeddings...')
  w2v = {}
  with open('data/vocab_embeddings.txt') as f:
    for line in f:
      line = line.split()
      word, emb = line[0], np.array(list(map(float, line[1:])))
      w2v[word] = emb
  return w2v


def avg_embedding(lyrics, w2v, dim):
  lyrics = nltk.word_tokenize(lyrics)
  embeddings = np.array([w2v.get(w, np.zeros(dim)) for w in lyrics])
  return embeddings.mean(axis=0)


if __name__ == '__main__':
  t0 = time.time()

  # Load data
  df_lyrics = load_lyrics()
  w2v = load_embeddings()

  # Get average embeddings
  df_lyrics['embedding'] = df_lyrics['lyrics'].apply(lambda l: avg_embedding(l, w2v, 300))

  # Save
  outfile = 'data/spotify_summary_0.30_embeddings.csv'
  print(f'Saving into {outfile}')
  df_lyrics[['artist', 'track', 'embedding']].to_csv(outfile, index=False)

  # Print time elapsed
  print(datetime.timedelta(seconds=time.time() - t0))
