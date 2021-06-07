import json
import pandas as pd
import sys

DATA_FOLDER = '../data'

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <lyrics_file.csv> <outputfile.csv>')
    sys.exit(-1)

  # Get songs with lyrics.
  df_lyrics = pd.read_csv(sys.argv[1])

  # Get spotify playlists.
  with open(f'{DATA_FOLDER}/spotify-million-playlist-challenge/challenge_set.json') as f:
    D = json.load(f)

  # Parse playlists.
  playlists_songs = []
  for p in D['playlists']:
    for t in p['tracks']:
      playlists_songs.append((p['pid'], t['artist_name'].lower(), t['track_name'].lower()))
  print(f'#songs in all playlists = {len(playlists_songs)}')

  # Convert to playlists dataframe.
  df_playlists = pd.DataFrame(playlists_songs, columns=['playlist_id', 'artist', 'track'])
  print(df_playlists.head())

  # Remove songs with no lyrics.
  df_playlists = df_playlists.merge(df_lyrics, on=['artist', 'track']).drop(columns=['lyrics'])

  # Save
  outfile = sys.argv[2]
  print(f'Saving into {outfile}')
  df_playlists.to_csv(outfile, index=False)
