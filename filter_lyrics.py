import pandas as pd
import numpy as np







def filter_duplicate(df, output_path):

    """

    This function removes duplicate songs

    It writes the result in output_path

    """

    df = df[['artist', 'track', 'lyrics']]

    df['rep_cnt'] = df.groupby(['artist', 'track']).transform(np.size)

    df_dups_no_linebreak = df.loc[(df['rep_cnt']==2) & (df['lyrics'].str.contains("\n") == False)] #duplicates with no line breaks

    df = pd.concat([df, df_dups_no_linebreak]).drop_duplicates(keep=False)[['artist', 'track', 'lyrics']] #no duplicates

    print("generating Spotify songs lyrics no duplicates ...")

    df[['artist', 'track', 'lyrics']].to_csv(output_path, index=False) #no duplicates, no Nan


def filter_linebreak_count(df, min_linebreak_count, output_path):
    """

    This function filters the lyrics that have less than min_linebreak_count number of linebreaks

    It writes the result in output_path

    """

    df['linebreakcount']=df.lyrics.str.count("\n")

    df = df.loc[df['linebreakcount']>=min_linebreak_count]


    print("generating Spotify songs only lyrics with at least {} linebreaks ...".format(min_linebreak_count))
    df[['artist', 'track', 'lyrics']].to_csv(output_path, index=False)   #only lyrics with min_linebreak_count line breaks


def main():
    df =pd.read_csv('data/spotify_songs_with_lyrics.csv') #non clean input, containing duplicates and NaN

    df = df.dropna()  #removing NaN

    min_linebreak_count=10

    output_path_for_mincount = 'data/spotify_lyrics_atLeast_{}linebreaks.csv'.format(min_linebreak_count)

    filter_linebreak_count(df, min_linebreak_count, output_path_for_mincount)

    output_path_for_noduplicate = 'data/spotify_lyrics_noduplicate.csv'
    filter_duplicate(df, output_path_for_noduplicate)



if __name__ == "__main__":
    main()
