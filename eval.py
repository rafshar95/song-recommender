import statistics
from math import log2
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import csv
import sys


def rank_songs(playlists_with_embeddings, dist, seed, index_id, num_recs, measure='max'):
    # measure can be {'avg', 'max'}
    songs = playlists_with_embeddings.drop_duplicates(['artist', 'track'], keep='last')

    for song in seed:
        songs = songs[songs['sid'] != song[index_id]]
    scores = []

    def compute_score(row):
        if measure == 'avg':
            a = statistics.mean([dist[row['sid']][song[index_id]] for song in seed])
            scores.append((a, row['sid']))

        elif measure == 'max':
            a = max([dist[row['sid']][song[index_id]] for song in seed])
            scores.append((a, row['sid']))

    songs.apply(compute_score, axis=1)
    sorted_songs = heapq.nlargest(num_recs, scores)
    filtered_songs = pd.DataFrame(columns=songs.columns)
    for i, j in sorted_songs:
        row = songs[songs['sid'] == j]
        filtered_songs = filtered_songs.append(row)
    return filtered_songs


def eval_playlist(ranked_songs, test, index_sid):
    # returns [recall@10, recall@50, NDCG@100]
    result = [0, 0, 0]

    rel = [False] * 100
    for i in test:
        found = (ranked_songs[:100]['sid'] == i[index_sid]).values.tolist()
        rel = [i or j for i, j in zip(found, rel)]

    # recall@10 & recall@50
    result[0] = sum(rel[:10]) / len(test)
    result[1] = sum(rel[:50]) / len(test)

    # NDCG@100
    dcg = sum([v/log2(i+2) for i, v in enumerate(rel)])
    ideal_rel = sorted(rel, reverse=True)
    idcg = sum([v/log2(i+2) for i, v in enumerate(ideal_rel)])
    if idcg == 0:
        result[2] = 0
    else:
        result[2] = dcg/idcg

    return result


def evaluate(playlists, dist, output_path, num_epochs=5, test_to_seed_ratio=0.2, num_recs=100):
    assert(100 <= num_recs <= playlists.shape[0])
    results = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
    index_sid = playlists.columns.get_loc('sid')

    for n in range(num_epochs):
        f = open(output_path + str(n) + ".csv", 'w', newline="")
        writer = csv.writer(f)
        step = 0
        for pid, p in playlists.groupby('playlist_id'):
            step += 1
            print("epoch " + str(n) + ", step " + str(step))
            if p.shape[0] <= 1:
                continue
            seed, test = train_test_split(p.values.tolist(), test_size=test_to_seed_ratio)
            ranked_songs = rank_songs(playlists, dist, seed, index_sid, num_recs)
            result = np.array(eval_playlist(ranked_songs, test, index_sid))
            results['pid'] += result
            writer.writerow(np.append(result, pid))
            f.flush()


def main():

    def embedding_proc(row):
        a = row['embedding']
        return list(map(float, a[1:len(a) - 1].split()))

    def dist_matrix(embeddings):
        a = np.array(embeddings)
        return cosine_similarity(a)

    df1 = pd.read_csv('data/' + sys.argv[1])
    df1['artist'] = df1['artist'].str.lower()
    df1['track'] = df1['track'].str.lower()

    df2 = pd.read_csv('data/' + sys.argv[2])
    df2['sid'] = range(df2.shape[0])
    df2['embedding'] = df2.apply(embedding_proc, axis=1)
    dist = dist_matrix(df2['embedding'].values.tolist())

    df3 = pd.merge(df2, df1, on=['artist', 'track'], how='left')

    evaluate(df3, dist, output_path="out/")


if __name__== "__main__":
    main()

