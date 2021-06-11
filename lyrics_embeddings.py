import numpy as np
import nltk
import pandas as pd
import time
import datetime
import sys
import torch
from transformers import BertTokenizer, BertModel
from torch.multiprocessing import Pool, current_process


PROCESSORS = 4


def load_lyrics(lyrics_file):
  print('Loading lyrics...')
  df = pd.read_csv(lyrics_file)
  df['lyrics'] = df['lyrics'].apply(str)
  return df


def w2v_embeddings(df_lyrics, agg_type, num_processors):
  print('Loading w2v embeddings...')
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
  return df_lyrics


def bert_worker(i, artist, track, lyrics, get_embedding_fn, agg_type, tokenizer, models):
  wid = int(current_process().name[-1]) - 1
  print(f'[processor {wid}] i={i:<6} {artist:<20} {track:>30}')
  embedding = get_embedding_fn(tokenizer, agg_type, models[wid], lyrics)
  with open(f'wid_{wid}_done.txt', 'a') as f:
    f.write(f'"{artist}","{track}","{embedding}"\n')
  return (artist, track, embedding)


def get_bert_embeddings(tokenizer, agg_type, model, lyrics):
  # Make new sentence every 10-th word.
  # sentences = [' '.join(s) for s in np.array_split(lyrics.split(), 10)]

  sentences = lyrics.split('\n')
  sentences = list(filter(None, sentences))  # Filter out empty lines
  sentences = [' '.join(s.split()[:369]) for s in sentences]

  # Tokenize and encode
  encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)

  # Get BERT predictions
  with torch.no_grad():
    output = model(**encoded_input)

  # Make layers a new dimension in tensor of hidden states
  hidden_states = torch.stack(output.hidden_states, dim=0)

  # Sum up embeddings of last 4 layers (apparently a good strategy i.e. see)
  # https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
  hidden_states = hidden_states[-4:, :, :, :].sum(dim=0)

  result_embedding = torch.zeros(hidden_states.shape[-1])

  if agg_type in ('avg', 'sum'):
    count_tok = 0
    for sent in range(hidden_states.shape[0]):
      for tok in range(hidden_states[sent].shape[0]):
        if encoded_input['attention_mask'][sent][tok] > 0:
          count_tok += 1
          result_embedding += hidden_states[sent][tok]
  else:
    raise Exception(f"Aggregation type '{agg_type}' not implemented!")

  if agg_type == 'avg':
    result_embedding /= count_tok

  return result_embedding.numpy()


def bert_embeddings(df_lyrics, agg_type, num_processors=PROCESSORS):
  print('Loading BERT model...')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  models = [
      BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
      for _ in range(num_processors)
  ]
  for model in models:
    model.eval()

  print('Computing BERT embeddings...')
  instances = [
      (i, row[0], row[1], row[2], get_bert_embeddings, agg_type, tokenizer, models)
      for i, row in enumerate(df_lyrics.values)
  ]

  with Pool(num_processors) as p:
    data = p.starmap(bert_worker, instances)

  return pd.DataFrame(data, columns=['artist', 'track', 'embedding'])


def parse_args(args):
  def exit():
    print(f'Usage: {args[0]} <lyrics_file.csv> <output_file.csv> <w2v|bert> <avg|sum> <num_processors>')
    sys.exit(-1)

  if len(args) != 6:
    exit()
  success = args[3] in ('w2v', 'bert')
  success &= args[4] in ('avg', 'sum')

  if not success:
    exit()
  return args[1:-1] + [int(args[-1])]


ADD_EMBEDDINGS_FN = {
    'w2v': w2v_embeddings,
    'bert': bert_embeddings
}


if __name__ == '__main__':
  lyrics_file, output_file, emb_type, agg_type, num_processors = parse_args(sys.argv)

  t0 = time.time()

  # Load data
  df_lyrics = load_lyrics(lyrics_file)

  # Add embeddings
  df_lyrics = ADD_EMBEDDINGS_FN[emb_type](df_lyrics, agg_type, num_processors)

  # Save
  print(f'Saving into {output_file}')
  df_lyrics[['artist', 'track', 'embedding']].to_csv(output_file, index=False)

  # Print time elapsed
  print(datetime.timedelta(seconds=time.time() - t0))
