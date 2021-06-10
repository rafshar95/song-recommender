import numpy as np
import nltk
import pandas as pd
import time
import datetime
import sys


def load_lyrics(lyrics_file):
  print('Loading lyrics...')
  df = pd.read_csv(lyrics_file)
  df['lyrics'] = df['lyrics'].apply(str)
  return df


def w2v_embeddings(df_lyrics, agg_type):
  print('Loading word embeddings...')
  w2v = {}
  with open('data/w2v_vocab_embeddings.txt') as f:
    for line in f:
      line = line.split()
      word, emb = line[0], np.array(list(map(float, line[1:])))
      w2v[word] = emb

  def _get_embedding(lyrics):
    lyrics = nltk.word_tokenize(lyrics)
    embeddings = np.array([w2v.get(w, np.zeros(300)) for w in lyrics])
    if agg_type == 'avg':
      return embeddings.mean(axis=0)
    elif agg_type == 'sum':
      return embeddings.sum(axis=0)
    raise Exception(f"Aggregation type '{agg_type}' not implemented!")

  # Get average embeddings
  df_lyrics['embedding'] = df_lyrics['lyrics'].apply(_get_embedding)


def parse_args(args):
  def exit():
    print(f'Usage: {args[0]} <lyrics_file.csv> <output_file.csv> <w2v|bert> <avg|sum|min|max>')
    sys.exit(-1)

  if len(args) != 5:
    exit()
  success = args[3] in ('w2v',)
  success &= args[4] in ('avg',)

  if not success:
    exit()
  return args[1:]


EMBEDDINGS_FN = {
    'w2v': w2v_embeddings
}


if __name__ == '__main__':
  lyrics_file, output_file, emb_type, agg_type = parse_args(sys.argv)

  t0 = time.time()

  # Load data
  df_lyrics = load_lyrics(lyrics_file)

  # Add embeddings
  EMBEDDINGS_FN[emb_type](df_lyrics, agg_type)

  # Save
  print(f'Saving into {output_file}')
  df_lyrics[['artist', 'track', 'embedding']].to_csv(output_file, index=False)

  # Print time elapsed
  print(datetime.timedelta(seconds=time.time() - t0))
