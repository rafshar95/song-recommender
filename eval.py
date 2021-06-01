import statistics
from math import log2
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import csv


def rank_songs(playlists_with_embeddings, dist, p_seed, index_id, num_recs, measure='max'):
    # measure can be {'avg', 'max'}
    songs = playlists_with_embeddings.drop_duplicates(['artist', 'track', 'embedding'], keep='last')
    for p in p_seed:
        songs = songs[songs['sid'] != p[index_id]]
    scores = []

    def compute_score(row):
        if measure == 'avg':
            a = statistics.mean([dist[row['sid']][seed[index_id]] for seed in p_seed])
            scores.append((a, row['sid']))

        elif measure == 'max':
            a = max([dist[row['sid']][seed[index_id]] for seed in p_seed])
            scores.append((a, row['sid']))
    songs.apply(compute_score, axis=1)
    sorted_songs = heapq.nlargest(num_recs, scores)
    filtered_songs = pd.DataFrame(columns=songs.columns)
    for i, j in sorted_songs:
        row = songs[songs['sid'] == j]
        filtered_songs = filtered_songs.append(row)
    return filtered_songs


def eval_playlist(ranked_songs, p_test, index_artist, index_track):
    # returns [precision@10, precision@50, NDCG@100]
    result = [0, 0, 0]
    ranked_songs.set_index(['artist', 'track'])

    rel = [False] * 100
    for i in p_test:
        found = ((ranked_songs[:100]['artist'] == i[index_artist]) & (ranked_songs[:100]['track'] == i[index_track])).values.tolist()
        rel = [i or j for i, j in zip(found, rel)]

    # precision@10 & precision@50
    result[0] = sum(rel[:10]) / len(p_test)
    result[1] = sum(rel[:50]) / len(p_test)

    # NDCG@100
    dcg = sum([v/log2(i+2) for i, v in enumerate(rel)])
    ideal_rel = sorted(rel, reverse=True)
    idcg = sum([v/log2(i+2) for i, v in enumerate(ideal_rel)])
    if idcg == 0:
        result[2] = 0
    else:
        result[2] = dcg/idcg

    return result


def evaluate(playlists_with_embeddings, dist, num_epochs=1, test_to_seed_ratio=0.2, num_recs=100):
    assert(100 <= num_recs <= playlists_with_embeddings.shape[0])

    # combine playlist and embedding data into a single dataframe
    # create new column to use when ranking songs
    playlists_with_embeddings['score'] = 0
    # holds scores (prec@10, prec@50, NDCG@100) for each playlist
    results = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
    index_artist = playlists_with_embeddings.columns.get_loc('artist')
    index_track = playlists_with_embeddings.columns.get_loc('track')
    index_id = playlists_with_embeddings.columns.get_loc('sid')
    seeds = [22, 90]
    for n in range(num_epochs):
        f = open('out/summ_epoch{}_frac{}.csv'.format(n, 0.05), 'w', newline="")
        writer = csv.writer(f)
        step = 0
        for pid, p in playlists_with_embeddings.groupby('playlist_id'):
            step+=1
            print("epoch " + str(n) + ", step " + str(step))
            if p.shape[0] <= 1 : continue
            p_seed, p_test = train_test_split(p.values.tolist(), test_size=test_to_seed_ratio, random_state=seeds[n])
            ranked_songs = rank_songs(playlists_with_embeddings, dist, p_seed, index_id, num_recs)
            result = np.array(eval_playlist(ranked_songs, p_test, index_artist, index_track))
            results['pid'] += result
            writer.writerow(np.append(result, pid))
            f.flush()
    # normalize scores by number of epochs
    results = [tuple([i/num_epochs for i in v]) for v in results.values()]
    # take average across all playlists
    results = [sum(i)/len(i) for i in zip(*results)]
    return results


def main():

    def embedding_proc(row):
        a = row['embedding']
        return list(map(float, a[1:len(a) - 1].split()))

    def dist_matrix(embeddings):
        a = np.array(embeddings)
        return cosine_similarity(a)

    df1 = pd.read_csv('data/spotify_playlists_with_embeddings_summ.csv')

    df2 = pd.read_csv('data/spotify_summary_0.05_embeddings.csv')
    df2['embedding'] = df2.apply(embedding_proc, axis=1)
    dist = dist_matrix(df2['embedding'].values.tolist())

    print(evaluate(df1, dist))

if __name__== "__main__":
    main()
