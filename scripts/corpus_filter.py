import argparse
import logging
from tqdm import tqdm
import numpy as np
from itertools import zip_longest

import worker_consumer as wc
import doc_similarity as ds


def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', type=str, required=True, help='Source language sentences')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target language sentences')
    parser.add_argument('-sc', '--scores', type=str, required=True, help='Initial scores')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-sim', '--similarities', type=str, required=True, help='Word similarity dictionary (embeddings)')
    parser.add_argument('-esim', '--edit_similarities', type=str, required=True, help='Word similarity dictionary (orthography)')
    parser.add_argument('-n', '--num_threads', default=1, type=int)
    parser.add_argument('-rs', '--remove_stopwords', default=1, type=int)
    parser.add_argument('-fl', '--from_language', type=str, default='de')
    parser.add_argument('-tl', '--to_language', type=str, default='en')
    parser.add_argument('-sw', '--similarity_weight', default=1.0, type=float)
    parser.add_argument('-ew', '--edit_similarity_weight', default=1.0, type=float)
    parser.add_argument('-m', '--method', default='maxalign', type=str, help='maxalign: segment detection based method; averaged word alignment scores')
    # These are only for max_align
    parser.add_argument('-mld', '--max_length_diff', default=None, type=int, help='Maximum length difference of aligned segments')
    parser.add_argument('-ml', '--min_length', default=0.7, type=float, help='Minimum length of detected parallel segments, otherwise sentence pair is not parallel (ratio of source sentence lenght)')
    parser.add_argument('-th', '--threshold', default=0.3, type=float, help='Similarity value threshold for segment detection')
    parser.add_argument('-ws', '--window_size', default=5, type=int, help='Average smooting window size')

    return parser.parse_args()


def helper(args, sims, fr_stopwords, to_stopwords, remove_stopwords,
           edit_similarities, similarity_weight, edit_similarity_weight,
           method, min_length=0.7, threshold=0.3, window_size=5,
           max_length_diff=5, **kwargs):
    s, t, sc = args

    if sc <= 0.0:
        return 0.0

    s = s.split()
    t = t.split()


    _, sim_score = method(s, (0, t), sims, fr_stopwords, to_stopwords,
            remove_stopwords=remove_stopwords, edit_similarities=edit_similarities,
            similarity_weight=similarity_weight, edit_similarity_weight=edit_similarity_weight,
            min_length=min_length, threshold=threshold, window_size=window_size,
            max_length_diff=max_length_diff
    )

    return sim_score


def reader(source, target, score):
    with open(source, 'r') as source_in, open(target, 'r') as target_in, open(score, 'r') as score_in:
        for s, t, sc in zip_longest(source_in, target_in, score_in, fillvalue=None):
            if s is None or t is None or sc is None:
                raise ValueError('Source, target and scores do not have the same amount of rows!')

            yield (s.strip(), t.strip(), float(sc.strip()))


def writer(x, output_stream, pbar, **kwargs):
    output_stream.write('{:.4f}\n'.format(x))
    output_stream.flush()
    pbar.update(1)


def main(source, target, scores, output, similarities, edit_similarities,
         from_language='de', to_language='en', num_threads=1, remove_stopwords=1,
         similarity_weight=1.0, edit_similarity_weight=1.0, method='maxalign',
         min_length=0.7, threshold=0.3, window_size=5, max_length_diff=5):
    remove_stopwords = remove_stopwords == 1

    sims = ds.load_similarity_file_todict(similarities)
    esims = ds.load_similarity_file_todict(edit_similarities)

    fr_stopwords = set(ds.stopwords.words(ds.lang_map[from_language]))
    to_stopwords = set(ds.stopwords.words(ds.lang_map[to_language]))

    methods = {
        'max': ds.max_similarity,
        'maxalign': ds.max_align,
    }
    method = methods[method]

    with open(output, 'w') as fout, tqdm() as pbar:
      wc.server(helper, writer, reader(source, target, scores),
                num_threads=num_threads, output_stream=fout, sims=sims, fr_stopwords=fr_stopwords,
                to_stopwords=to_stopwords, remove_stopwords=remove_stopwords,
                edit_similarities=esims, pbar=pbar, similarity_weight=similarity_weight,
                edit_similarity_weight=edit_similarity_weight, method=method,
                min_length=min_length, threshold=threshold, window_size=window_size,
                max_length_diff=max_length_diff)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed = 0

    args = getArguments()

    main(**vars(args))
