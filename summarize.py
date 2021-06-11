import pandas as pd
import time
import datetime
import nltk


#from transformers import GPT2Tokenizer,GPT2LMHeadModel
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
#my_gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#my_gpt2model =  GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=my_gpt2tokenizer.eos_token_id)


cnt = 0

def t5_summarize(original_text, fraction):
    #print("1")
    """
    cnt+=1

    if(cnt%10==0):
        print(cnt)
    """
    global cnt

    cnt = cnt +1

    print(cnt)


    text = "summarize:" + original_text

    input_ids=tokenizer.encode(text, return_tensors='pt', max_length = 2048, truncation = True, padding = True)
    #print("input_ids:", input_ids, len(input_ids[0]), len(text.split()))
    #print(len(input_ids[0]))
    f = fraction
    summary_ids = my_model.generate(input_ids, min_length = int(f * len(input_ids[0])), max_length =  int(f * len(input_ids[0])))

    #print(len(summary_ids[0]))

    t5_summary = tokenizer.decode(summary_ids[0],  skip_special_tokens=True)

    return t5_summary



def gpt2_summarize(original_text, fraction):
    # Importing model and tokenizer


    # Instantiating the model and tokenizer with gpt-2
    #tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    #model=GPT2LMHeadModel.from_pretrained('gpt2')

    text = "summarize:" + original_text

    # Encoding text to get input ids & pass them to model.generate()
    input_ids=my_gpt2tokenizer.encode(text,return_tensors='pt',max_length=2048)
    f = fraction
    summary_ids=my_gpt2model.generate(input_ids,early_stopping=True)
    # Decoding and printing summary

    GPT2_summary=my_gpt2tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    return GPT2_summary

def lsa_summarize(original_text):

    from sumy.summarizers.lsa import LsaSummarizer

    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

    lsa_summarizer=LsaSummarizer()
    lsa_summary= lsa_summarizer(parser.document,5)

    # Printing the summary
    for sentence in lsa_summary:
        print(sentence)

def luhn_summarize(original_text):
    from sumy.summarizers.luhn import LuhnSummarizer

    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

    luhn_summarizer=LuhnSummarizer()
    luhn_summary=luhn_summarizer(parser.document,sentences_count=5)

    # Printing the summary
    for sentence in luhn_summary:
        print(sentence)



def lexRank_summarize(original_text):
    import sumy

    """
    in  a python sell:
    import nltk
    nltk.download('punkt')
    """

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer

    from sumy.summarizers.lex_rank import LexRankSummarizer

    myParser = PlaintextParser.from_string(original_text,Tokenizer('english'))

    # Creating a summary of 3 sentences.
    lexRank_summarizer = LexRankSummarizer()
    summary= lexRank_summarizer(myParser.document,sentences_count=5)
    for sentence in summary:
        print(sentence)



def gensim_summarize(original_text):


    """
        try pip install numpy==1.18.0
        try pip install scipy 1.1.0
        pip install gensim==3.8.3
        print(sc.__version__)
        print(np.__version__)
        with open('data/lyrics.txt') as f:
            original_text = f.read()

    """
    import gensim
    from gensim.summarization import summarize
    print(summarize(original_text, word_count =  50)) #or ratio

def klsum_summarize(original_text, s):

    global cnt

    cnt +=1

    if(cnt%100==0):
        print(cnt)


    from sumy.summarizers.kl import KLSummarizer

    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(original_text,Tokenizer('english'))

    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(parser.document,sentences_count=s)

    # Printing the summary
    res = ""
    for sentence in kl_summary:

        res+= (str(sentence) + '.' + "\n")

    return res


def load_lyrics(infile):

    print('Loading lyrics...')
    df = pd.read_csv(infile)
    df['lyrics'] = df['lyrics'].apply(str)
    return df


def main():
    df_lyrics = load_lyrics
    t0 = time.time()
    # Load data

    """

    for f in range(3, 4):

        infile = 'data/spotify_lyrics.csv'
        df_lyrics = load_lyrics(infile)

        df_lyrics['lyrics'] = df_lyrics['lyrics'].apply(lambda l: t5_summarize(l, fraction = f/10))
        # Save
        outfile = 'data/spotify_summaries_T5_{}.csv'.format(f*10)
        print(f'Saving into {outfile}')
        df_lyrics.to_csv(outfile, index=False)
        """

    for s in range(2,3):

        infile = 'data/spotify_lyrics_atLeast_10linebreaks.csv'
        df_lyrics = load_lyrics(infile)

        df_lyrics['lyrics'] = df_lyrics['lyrics'].apply(lambda l: klsum_summarize(l, s))

        # Save
        outfile = 'data/spotify_summaries_klsum_{}sentences.csv'.format(s)
        print(f'Saving into {outfile}')
        df_lyrics.to_csv(outfile, index=False)




    print(datetime.timedelta(seconds=time.time() - t0))

    """
    with open('data/lyrics.txt') as f:
        original_text = f.read().replace('\n', ". \n")

    """
    #print(original_text)
    #t5_summarize(original_text, 0.5)

if __name__ == "__main__":
    main()
