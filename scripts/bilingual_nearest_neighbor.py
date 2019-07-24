import argparse
import logging
from gensim.models import KeyedVectors
import numpy as np
import faiss
from tqdm import tqdm


def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-se', '--source_embeddings', type=str, required=True)
    parser.add_argument('-te', '--target_embeddings', type=str, default=None)
    parser.add_argument('-b', '--binary', type=int, default=0, help='Input embedding format')
    parser.add_argument('-k', '--knn', type=int, default=10, help='k most similar target words in the dictionary')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-mv', '--max_vocab', type=int, default=None, help='Limits the number of words in the source and target vocabulary respectively')
    parser.add_argument('--float_precision', type=int, default=4, help='Output format')
    parser.add_argument('-m', '--method', type=str, default='nn', help='nn: cosine, csls: CSLS')
    parser.add_argument('--cslsknn', type=int, default=10, help='knn for csls; knn <= cslsknn')
    parser.add_argument('--gpu', type=int, default=1, help='0: CPU; 1: GPU')

    return parser.parse_args()


def get_index(vector_size, gpu=True):
    if gpu and hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0

        index = faiss.GpuIndexFlatIP(res, vector_size, config)

        return index, res, config # objects other than index are needed to prevent error due to GC
    else:
        index = faiss.IndexFlatIP(vector_size)
        return index, None, None


def get_embeddings_as_array(w2v):
    res = np.ones((len(w2v.vocab), w2v.vector_size)).astype('float32')

    for i, w in enumerate(w2v.vocab):
        res[i] = w2v[w]/np.linalg.norm(w2v[w])

    return res


def get_nn(se, te, knn, gpu=True):
    embeddings = get_embeddings_as_array(te)

    index, res, config = get_index(se.vector_size, gpu)
    index.add(embeddings)

    logging.warning('Running search query...')
    return index.search(get_embeddings_as_array(se), knn)


def get_csls(se, te, knn, csls_knn, gpu=True):
    assert knn <= csls_knn

    st_distances, st_indices = get_nn(se, te, csls_knn, gpu)
    ts_distances, ts_indices = get_nn(te, se, csls_knn, gpu)

    res_distances = np.zeros((st_distances.shape[0], knn), dtype=st_distances.dtype)
    res_indices = np.zeros((st_distances.shape[0], knn), dtype=st_indices.dtype)

    logging.warning('Calculating CSLS...')
    rs_lst = [dists.mean() for dists in ts_distances]

    for sidx, sindices in enumerate(tqdm(st_indices)):
        rt = st_distances[sidx].mean()
        csls_dists = list()

        for ti, tidx in enumerate(sindices):
            rs = rs_lst[tidx]
            csls_dists.append(2*st_distances[sidx][ti] - rt - rs)

        csls_dists = np.array(csls_dists)
        max_idxs = np.argsort(csls_dists)[::-1][:knn]
        res_distances[sidx] = csls_dists[max_idxs]
        res_indices[sidx] = sindices[max_idxs]

    return res_distances, res_indices


def main(source_embeddings, binary, output, target_embeddings=None, knn=10,
         max_vocab=None, float_precision=4, method='nn', cslsknn=10, gpu=True):
    method = method.split('-')
    method = method[0]

    se = KeyedVectors.load_word2vec_format(source_embeddings, binary=binary, unicode_errors='replace', limit=max_vocab)
    if target_embeddings is None or target_embeddings == source_embeddings:
        te = se
    else:
        te = KeyedVectors.load_word2vec_format(target_embeddings, binary=binary, unicode_errors='replace', limit=max_vocab)

    if method == 'nn':
        distances, indices = get_nn(se, te, knn, gpu)
    elif method == 'csls':
        distances, indices = get_csls(se, te, knn, cslsknn, gpu)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    logging.warning('Saving results...')
    with open(output, 'w') as fout:
        for i, idxs in enumerate(indices):
            sw = se.index2word[i]
            tmp = ['{}\t{}'.format(te.index2word[idxs[j]], format(distances[i][j], '.{}f'.format(float_precision))) for j in range(knn)]

            fout.write('{}\t{}\n'.format(sw, '\t'.join(tmp)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = getArguments()

    main(**vars(args))
