import pandas as pd
import numpy as np


df =pd.read_csv('data/spotify_songs_with_lyrics.csv') #non clean input, containing duplicates and NaN



df = df.dropna()  #removing NaN

df['size'] = df.groupby(['artist', 'track']).transform(np.size)



df_only_with_linebreaks = df.loc[df['lyrics'].str.contains("\n")== True] #containing only the lyrics with line breaks


print("generating Spotify songs only lyrics with linebreaks ...")


path_for_lyrics_with_linebreaks = 'data/spotify_songs_with_lyrics_linebreaks.csv'
df_only_with_linebreaks[['artist', 'track', 'lyrics']].to_csv(path_for_lyrics_with_linebreaks, index=False)   #only lyrics with line breaks


print(df_only_with_linebreaks.shape[0])


df_dups_no_break = df.loc[(df['size']==2) & (df['lyrics'].str.contains("\n") == False)] #duplicates with no line breaks


print(df_dups_no_break.shape[0])


df_no_dups= pd.concat([df, df_dups_no_break]).drop_duplicates(keep=False)[['artist', 'track', 'lyrics']] #no duplicates


print("generating Spotify songs lyrics no duplicates ...")

path_for_lyrics_no_duplicates = 'data/spotify_songs_with_lyrics_no_duplicates.csv'
df_no_dups[['artist', 'track', 'lyrics']].to_csv(path_for_lyrics_no_duplicates, index=False) #no duplicates, no Nan


print(df_no_dups.shape[0])
