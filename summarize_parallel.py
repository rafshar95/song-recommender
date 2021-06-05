import pandas as pd
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

import time
import datetime

import numpy as np
from multiprocessing import Pool
import os
import sys

PROCESSORS = 4
T5 = T5ForConditionalGeneration.from_pretrained('t5-small')
TOKENIZER = T5Tokenizer.from_pretrained('t5-small')

# def gpt2_summarize(original_text):
#     # Importing model and tokenizer
#     from transformers import GPT2Tokenizer,GPT2LMHeadModel

#     # Instantiating the model and tokenizer with gpt-2
#     tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
#     model=GPT2LMHeadModel.from_pretrained('gpt2')

#     # Encoding text to get input ids & pass them to model.generate()
#     inputs=tokenizer.batch_encode_plus([original_text],return_tensors='pt',max_length=512)
#     summary_ids=model.generate(inputs['input_ids'],early_stopping=True)
#     # Decoding and printing summary

#     GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
#     print(GPT_summary)

# def lsa_summarize(original_text):

#     from sumy.summarizers.lsa import LsaSummarizer

#     from sumy.nlp.tokenizers import Tokenizer
#     from sumy.parsers.plaintext import PlaintextParser
#     parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

#     lsa_summarizer=LsaSummarizer()
#     lsa_summary= lsa_summarizer(parser.document,5)

#     # Printing the summary
#     for sentence in lsa_summary:
#         print(sentence)

# def luhn_summarize(original_text):
#     from sumy.summarizers.luhn import LuhnSummarizer

#     from sumy.nlp.tokenizers import Tokenizer
#     from sumy.parsers.plaintext import PlaintextParser
#     parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

#     luhn_summarizer=LuhnSummarizer()
#     luhn_summary=luhn_summarizer(parser.document,sentences_count=5)

#     # Printing the summary
#     for sentence in luhn_summary:
#         print(sentence)



# def lexRank_summarize(original_text):
#     import sumy

#     """
#     in  a python sell:
#     import nltk
#     nltk.download('punkt')
#     """

#     from sumy.parsers.plaintext import PlaintextParser
#     from sumy.nlp.tokenizers import Tokenizer

#     from sumy.summarizers.lex_rank import LexRankSummarizer

#     myParser = PlaintextParser.from_string(original_text,Tokenizer('english'))

#     # Creating a summary of 3 sentences.
#     lexRank_summarizer = LexRankSummarizer()
#     summary= lexRank_summarizer(myParser.document,sentences_count=5)
#     for sentence in summary:
#         print(sentence)


# def gensim_summarize(original_text):


#     """
#         try pip install numpy==1.18.0
#         try pip install scipy 1.1.0
#         pip install gensim==3.8.3
#         print(sc.__version__)
#         print(np.__version__)
#         with open('data/lyrics.txt') as f:
#             original_text = f.read()
#     """
#     import gensim
#     from gensim.summarization import summarize
#     print(summarize(original_text, word_count =  50)) #or ratio


# def klsum_summarize(original_text):
#     from sumy.summarizers.kl import KLSummarizer

#     from sumy.nlp.tokenizers import Tokenizer
#     from sumy.parsers.plaintext import PlaintextParser
#     parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

#     kl_summarizer=KLSummarizer()
#     kl_summary=kl_summarizer(parser.document,sentences_count=5)

#     # Printing the summary
#     for sentence in kl_summary:
#         print(sentence)


def t5_summarize(original_text, fraction):
    #print("1")
    """
    cnt+=1

    if(cnt%10==0):
        print(cnt)
    """

    text = "summarize:" + original_text

    # print('Enconding...')
    input_ids=TOKENIZER.encode(text, return_tensors='pt', max_length = 2048, truncation = True, padding = True)
    #print("input_ids:", input_ids, len(input_ids[0]), len(text.split()))
    #print(len(input_ids[0]))

    f = fraction
    # print('Generating...')
    summary_ids = T5.generate(input_ids, min_length = int(f * len(input_ids[0])), max_length =  int(f * len(input_ids[0])))

    #print(len(summary_ids[0]))

    # print('Decoding...')
    t5_summary = TOKENIZER.decode(summary_ids[0],  skip_special_tokens=True)

    return t5_summary


def load_lyrics(path):
    print('Loading lyrics...')
    df = pd.read_csv(path)
    df['lyrics'] = df['lyrics'].apply(str)
    return df


def worker(worker_id, lyrics_batch, fraction):
  def save(rows):
    df = pd.DataFrame(rows, columns=['artist', 'track', 'lyrics'])
    df.to_csv(f'dataset/data/spotify_summaries__{worker_id}__{fraction:.2f}__missing.csv', index=False)

  summaries = []
  i = 0
  for artist, track, lyrics in lyrics_batch:
    print(f'wid={worker_id} i={i}')
    summaries.append((artist, track, t5_summarize(lyrics, fraction)))

    if i % 10 == 0:
      print(f'wid={worker_id} saving...')
      save(summaries)

    i += 1

  save(summaries)


def main():
  t0 = time.time()

  df = load_lyrics(sys.argv[1])

  for fraction in [0.3]:
    worker_ids = list(range(PROCESSORS))
    lyrics_batches = np.array_split(df.values, PROCESSORS)
    fractions = [fraction] * PROCESSORS

    workers_instances = list(zip(worker_ids, lyrics_batches, fractions))

    with Pool(processes=PROCESSORS) as pool:
      pool.starmap(worker, workers_instances)

    with open(f'{fraction}.txt', 'w') as f:
      f.write(f"{fraction} done\n")
  print(datetime.timedelta(seconds=time.time() - t0))



if __name__ == "__main__":
  main()
