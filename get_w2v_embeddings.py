from gensim.models import KeyedVectors
from gensim.downloader import base_dir
import os
import pandas as pd
import time
import datetime


def load_data():
  path = os.path.join(base_dir, 'word2vec-google-news-300', "word2vec-google-news-300.gz")
  model = KeyedVectors.load_word2vec_format(path, binary=True)
  return model


def vocabulary(path, lyrics_col=2):
  df = pd.read_csv(path)
  V = []
  for row in df.values:
    V += str(row[lyrics_col]).split()
  return set(V)


if __name__ == '__main__':
  t0 = time.time()

  # Get vocabulary.
  V = vocabulary('data/spotify_songs_with_lyrics.csv')

  # Load google news w2v...
  print(f'Loading google news word2vec...')
  google_w2v = load_data()

  # ... takes a while, so saving compressed version in file,
  # which contains embeddings for words in vocabulary.
  outfile = 'data/w2v_vocab_embeddings.txt'
  print(f'Saving w2v vocabulary embeddings into {outfile}')
  with open(outfile, 'w') as f:
    for word in V:
      if word in google_w2v:
        f.write(f"{word} {' '.join(map(str, google_w2v[word]))}\n")

  # Print time elapsed
  print(datetime.timedelta(seconds=time.time() - t0))
