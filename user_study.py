import pandas as pd
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from time import sleep
import sys


df1 = pd.read_csv('data/'+sys.argv[1])
df1['artist'] = df1['artist'].str.lower()
df1['track'] = df1['track'].str.lower()

df2 = pd.read_csv('data/'+sys.argv[2])
df2['sid'] = range(df2.shape[0])

df_playlists = pd.merge(df2, df1, on=['artist', 'track'], how='left')

df3_all = pd.merge(pd.read_csv('data/'+sys.argv[3]), df2, on=['artist', 'track'], how='left').drop('embedding', 1)

df3_summ = pd.merge(pd.read_csv('data/'+sys.argv[4]), df2, on=['artist', 'track'], how='left').drop('embedding', 1)

num_songs = df2.shape[0]

playlists = defaultdict(list)
for pid, p in df_playlists.groupby('playlist_id'):
    playlists[p.shape[0]] += [pid]

p_len = -1
while not playlists[p_len]:
    p_len = int(input("Enter playlist length. If playlist with given length does not exist, prompt will ask again: "))
playlist = list(df_playlists.loc[df_playlists['playlist_id'] == random.choice(playlists[p_len])]['sid'])

# s_len = p_len
# while s_len >= p_len or s_len < 1:
#     s_len = int(input("Enter seed length (must be less than playlist length): "))
# seed, test = train_test_split(playlist, train_size=s_len)
seed, test = train_test_split(playlist, train_size=1)

random_len = num_songs
while random_len > num_songs-len(test) or random_len < 1:
    random_len = int(input("Enter number of random songs you would like to add (must be at least 1): "))
random_sids = random.sample([song for song in range(num_songs) if song not in playlist], random_len)
test_with_randoms = test + random_sids
random.shuffle(test_with_randoms)

print("Starting part 1")
sleep(1)
score1 = 0
print("Seed songs:")
for s in seed:
    print(df3_all.loc[df3_all['sid'] == s]['lyrics'].values[0])
print("*" * 100)
print("*" * 100)
print("*" * 100)
print("Recommend new songs:")
for song in test_with_randoms:
    print(f"Song {test_with_randoms.index(song)+1}:")
    print(df3_all.loc[df3_all['sid'] == song]['lyrics'].values[0])
    prediction = int(input("\nEnter 1 if you recommend this song, 0 if you do not: "))
    if prediction == (song in playlist):
        score1 += 1
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)

sleep(1)
score2 = 0
print("Seed song:")
for s in seed:
    print(df3_summ.loc[df3_summ['sid'] == s]['lyrics'].values[0])
print("*"*100)
print("*"*100)
print("*"*100)
for song in test_with_randoms:
    print(f"Song {test_with_randoms.index(song)+1}:")
    print(df3_summ.loc[df3_summ['sid'] == song]['lyrics'].values[0])
    prediction = int(input("\nEnter 1 if you recommend this song, 0 if you do not: "))
    if prediction == (song in playlist):
        score2 += 1
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
print(f"Score for part 1: {score1/len(test_with_randoms):.2f} Score for part 2: {score2/len(test_with_randoms):.2f}")
