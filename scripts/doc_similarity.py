import argparse
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
import re
from itertools import zip_longest

from nltk.corpus import stopwords

import worker_consumer as wc


general_stopwords = {'&quot;', '&amp;', '&apos;', '&gt;', '&lt;', '&quot;'}.union(set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

lang_map = {
    'en': 'english',
    'de': 'german',
    'fr': 'french',
    'ru': 'russian'
}

digits_pattern = re.compile('[0-9]+([,][0-9]+)?([.][0-9]+)?')


def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', type=str, required=True, help='Source language sentences. Format: <ID>\t<sentence>')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target language sentences')
    parser.add_argument('-o', '--output', type=str, required=True, help='Format: <SRC_ID>\t(<TRG_ID>\t<similarity>)*')
    parser.add_argument('-sim', '--similarities', type=str, required=True, help='Word similarity dictionary (embeddings)')
    parser.add_argument('-dsim', '--document_similarities', type=str, required=True, help='Document similarities for prefiltering')
    parser.add_argument('-esim', '--edit_similarities', type=str, default=None, help='Word similarity dictionary (orthography)')
    parser.add_argument('-n', '--num_threads', default=1, type=int)
    parser.add_argument('-k', '--knn', default=5, type=int, help='Save k most similar sentences')
    parser.add_argument('-rs', '--remove_stopwords', default=1, type=int)
    parser.add_argument('-fl', '--from_language', type=str, required=True, help='Source language (for stopword filtering)')
    parser.add_argument('-tl', '--to_language', type=str, default='en', help='Target language')
    parser.add_argument('-m', '--method', default='maxalign', type=str, help='maxalign: segment detection based method; averaged word alignment scores')
    # These are only for max_align
    parser.add_argument('-mld', '--max_length_diff', default=None, type=int, help='Maximum length difference of aligned segments')
    parser.add_argument('-ml', '--min_length', default=0.7, type=float, help='Minimum length of detected parallel segments, otherwise sentence pair is not parallel (ratio of source sentence lenght)')
    parser.add_argument('-th', '--threshold', default=0.3, type=float, help='Similarity value threshold for segment detection')
    parser.add_argument('-ws', '--window_size', default=5, type=int, help='Average smooting window size')

    return parser.parse_args()


def load_data(file):
    logger.info('Reading file: {}'.format(file))
    res = dict()
    with open(file, 'r') as fin:
        for line in fin:
            id, text = line.split('\t')
            res[id] = text.split()
    return res


def load_similarity_file(file):
    res = dict()

    with open(file, 'r') as fin:
        for line in fin:
            data = line.split('\t')
            word = data[0]

            if word in res:
                logger.warning('Word {} is already seen. Ignoring!'.format(word))
                continue

            res[word] = [(data[i*2 + 1], float(data[i*2 + 2])) for i in range(int((len(data)-1)/2))]

    return res


def load_similarity_file_todict(file):
    res = dict()

    with open(file, 'r') as fin:
        for line in fin:
            data = line.split('\t')
            word = data[0]

            if word in res:
                logger.warning('Word {} is already seen. Ignoring!'.format(word))
                continue

            res[word] = {data[i*2 + 1]: float(data[i*2 + 2]) for i in range(int((len(data)-1)/2))}

    return res


class SmartSimilarity():

    def __init__(self, sim_file, ignore_digits=False, exact_digit_match=False,
                 dict_format=True):
        assert not ignore_digits or not exact_digit_match

        self.ignore_digits = ignore_digits
        self.exact_digit_match = exact_digit_match
        self.sims = load_similarity_file_todict(sim_file) if dict_format else load_similarity_file(sim_file)

    def __getitem__(self, key):
        if self.ignore_digits or self.exact_digit_match:
            digit = digits_pattern.match(key) is not None
            if self.ignore_digits and digit:
                return dict() if type(self.sims) == dict else list()
            elif digit:
                return {key: 1.0} if type(self.sims) == dict else [(key, 1.0)]
            else:
                return self.sims[key]
        else:
            return self.sims[key]

    def __contains__(self, key):
        if self.ignore_digits or self.exact_digit_match:
            digit = digits_pattern.match(key) is not None
            if self.ignore_digits and digit:
                return False
            elif digit:
                return True
            else:
                return key in self.sims
        else:
            return key in self.sims


def get_similarity(s, t, sim):
    if s not in sim:
        s = s.lower()
    if s not in sim:
        return None
    if t not in sim[s]:
        t = t.lower()
    if t not in sim[s]:
        return None

    return sim[s][t]


def get_max_word_similarity(source, target, word_sims=None, edit_sims=None,
                            word_sims_weight=1.0, edit_sims_weight=1.0):
    assert word_sims or edit_sims

    wsims = None
    if word_sims and word_sims_weight != 0.0:
        wsim = get_similarity(source, target, word_sims)
        if wsim is not None:
            wsim *= word_sims_weight

    esim = None
    if edit_sims is not None and edit_sims_weight != 0.0:
        esim = get_similarity(source, target, edit_sims)
        if esim is not None:
            esim *= edit_sims_weight

    if wsim is None and esim is None:
        return None
    elif wsim is None:
        return esim
    elif esim is None:
        return wsim

    return max(wsim, esim)


def get_max_alignment(src, tgt, sim, edit_similarities=None, stopwords=None,
                      similarity_weight=1.0, edit_similarity_weight=1.0,
                      remove_paired = True):

    res = np.zeros((len(src), len(tgt)))
    seen = set()
    for sidx, s in enumerate(src):
        max_sim = -1.0
        max_idx = -1
        if stopwords is None or s not in stopwords[0]:
            for idx, t in enumerate(tgt):
                if idx in seen or (stopwords and t in stopwords[1]):
                    continue

                tmp_sim = get_max_word_similarity(s, t, sim, edit_similarities,
                                                    similarity_weight, edit_similarity_weight)
                if tmp_sim is None:
                    continue

                if tmp_sim > max_sim:
                    max_sim = tmp_sim
                    max_idx = idx

        if max_idx > -1:
            res[sidx][max_idx] = max_sim
            if remove_paired:
                seen.add(max_idx)

    return res


def max_similarity(src, tgt, sim, fr_stopwords, to_stopwords,
                   remove_stopwords=True, edit_similarities=None,
                   similarity_weight=1.0, edit_similarity_weight=1.0,
                   remove_paired = True, **kwargs):

    l_fr_stopwords = general_stopwords
    l_to_stopwords = general_stopwords
    if remove_stopwords:
        l_fr_stopwords = l_fr_stopwords.union(fr_stopwords)
        l_to_stopwords = l_to_stopwords.union(to_stopwords)

    src = [w for w in src if w not in l_fr_stopwords and ( not remove_stopwords or digits_pattern.search(w) is None)]
    text = [w for w in tgt[1] if w not in l_to_stopwords and ( not remove_stopwords or digits_pattern.search(w) is None)]

    N = len(text)

    tmp = list()
    for s in src:
        max_sim = -1.0
        max_idx = -1
        for idx, t in enumerate(text):
            tmp_sim = get_max_word_similarity(s, t, sim, edit_similarities,
                                            similarity_weight, edit_similarity_weight)

            if tmp_sim is None:
                continue

            if tmp_sim > max_sim:
              max_sim = tmp_sim
              max_idx = idx

        if max_idx > -1:
          tmp.append(max_sim)
          if remove_paired:
             del text[max_idx]

    if len(tmp) == 0:
        tmp = 0.0
    else:
        tmp = np.sum(tmp)

    if N == 0:
        N = 1.0
    return (tgt[0], tmp/N)


def get_subsentences(scores, threshold=0.5, min_length=5):
    res = list()
    start = None
    for i, s in enumerate(scores):
        if start is None and s >= threshold:
            start = i
        elif start is not None and s < threshold:
            if i-start >=min_length:
                res.append((start, i))
            start = None
    else:
        if start is not None:
            n = len(scores)
            if n-start >=min_length:
                res.append((start, n))

    return res


def get_averaged_alignment_scores(alignment, window_size=5):
    if window_size % 2 == 1:
        window_size -= 1
    window_size = int(window_size/2)

    n = len(alignment)
    res = [np.mean(alignment[max(0, i-window_size):min(n, i+window_size)]) for i in range(n)]

    return res


def get_parallel_subsentences(s2t_alignments, t2s_alignments=None, window_size=5,
                              threshold=0.5, min_length=5, max_length_diff=None):
    s2t_align_scores = s2t_alignments.max(axis=1)
    if t2s_alignments is None:
        t2s_alignments = s2t_alignments.transpose()
    t2s_align_scores = t2s_alignments.max(axis=1)

    s2t_scores = get_averaged_alignment_scores(s2t_align_scores, window_size=window_size)
    ssubs = get_subsentences(s2t_scores, threshold, min_length)
    t2s_scores = get_averaged_alignment_scores(t2s_align_scores, window_size=window_size)
    tsubs = get_subsentences(t2s_scores, threshold, min_length)

    res = list()

    for si, ssub in enumerate(ssubs):
        max_idx = -1
        max_match = -1
        if len(tsubs) == 0:
            break
        for ti, tsub in enumerate(tsubs):
            if max_length_diff is not None and abs(ssub[1]-ssub[0] - (tsub[1]-tsub[0])) > max_length_diff:
                continue
            tmp_match = sum([tsub[0] <= i < tsub[1] for i in range(ssub[0], ssub[1])])
            if tmp_match > max_match:
                max_idx = ti
                max_match = tmp_match

        if max_idx > -1:
            res.append((ssub, tsubs.pop(max_idx)))

    return res


def max_align(src, tgt, sim, fr_stopwords, to_stopwords, remove_stopwords=True,
              edit_similarities=None, similarity_weight=1.0, edit_similarity_weight=1.0,
              remove_paired = True, min_length=0.7, threshold=0.3, window_size=5,
              max_length_diff=5):

    l_fr_stopwords = general_stopwords
    l_to_stopwords = general_stopwords
    if remove_stopwords:
        l_fr_stopwords = l_fr_stopwords.union(fr_stopwords)
        l_to_stopwords = l_to_stopwords.union(to_stopwords)
    target_text = tgt[1]

    a = get_max_alignment(src, target_text, sim, edit_similarities, None,
                            similarity_weight, edit_similarity_weight, remove_paired)

    score = a.sum()/a.shape[1]
    if 0.0 < min_length < 1.0:
        min_length = int(min(len(src), len(target_text))*min_length)

    subs = get_parallel_subsentences(a, min_length=min_length, threshold=threshold,
                                    window_size=window_size, max_length_diff=max_length_diff)
    sub_len = float(max([sub[0][1]-sub[0][0] for sub in subs]+[0]))

    res = score * (sub_len/len(src))
    return (tgt[0], res)


def helper(args, sims, fr_stopwords, to_stopwords, candidates, method,
           remove_stopwords, edit_similarities, max_length_diff, knn=5,
           min_length=0.7, threshold=0.3, window_size=5, **kwargs):
    (id, text), tids = args
    text = text.split()
    len_text = len(text)
    return (id, list(sorted(
        [
            method(text, t, sims, fr_stopwords, to_stopwords,
                   remove_stopwords=remove_stopwords,
                   edit_similarities=edit_similarities, min_length=min_length,
                   threshold=threshold, window_size=window_size, max_length_diff=max_length_diff
            )
            for t in [(tid, candidates[tid]) for tid in tids]
            if max_length_diff is None or abs(len_text-len(t[1])) <= max_length_diff],
        key=lambda x: x[1], reverse=True))[:knn]
    )


def reader(source, document_similarities):
    with open(source, 'r') as sin, open(document_similarities, 'r') as din:
        for sline, dline in zip_longest(sin, din, fillvalue=None):
            if sline is None or dline is None:
                raise ValueError('Source and document similarity files have unmatched length!')

            sline = sline.strip().split('\t')
            dline = dline.strip().split('\t')
            if sline[0] != dline[0]:
                raise ValueError('Lines in source and document similarity files do not match: src.id {} - doc.id {}'.format(sline[0], dline[0]))

            yield (sline, [dline[i] for i in range(1, len(dline), 2)])


def writer(x, output_stream, pbar, **kwargs):
    id, sim = x
    sim = [item for item in sim if item[1] > 0.0]
    if len(sim) > 0:
        output_stream.write('{}\t{}\n'.format(id, '\t'.join(['{}\t{:10.4f}'.format(item[0], item[1]) for item in sim])))
        output_stream.flush()
    pbar.update(1)


def main(source, target, output, similarities, document_similarities, from_language,
         to_language='en', method='maxalign', num_threads=1, knn=5, remove_stopwords=1,
         edit_similarities=None, max_length_diff=None, min_length=0.7, threshold=0.3, window_size=5):
    remove_stopwords = remove_stopwords == 1
    if max_length_diff is not None and max_length_diff < 1:
        max_length_diff = None

    logger.warning('Loading word similarities...')
    sims = SmartSimilarity(similarities, ignore_digits=(edit_similarities is not None))
    esims = None
    if edit_similarities is not None:
        logger.warning('Loading Levenstein similarities...')
        esims = SmartSimilarity(edit_similarities, exact_digit_match=True)
    candidates = load_data(target)

    fr_stopwords = set(stopwords.words(lang_map[from_language]))
    to_stopwords = set(stopwords.words(lang_map[to_language]))

    methods = {
        'max': max_similarity,
        'maxalign': max_align,
    }
    method = methods[method]

    with open(output, 'w') as fout, tqdm() as pbar:
        wc.server(helper, writer, reader(source, document_similarities), num_threads=num_threads,
            output_stream=fout, sims=sims, fr_stopwords=fr_stopwords,
            to_stopwords=to_stopwords, candidates=candidates, method=method,
            remove_stopwords=remove_stopwords, edit_similarities=esims,
            pbar=pbar, max_length_diff=max_length_diff, knn=knn,
            min_length=min_length, threshold=threshold, window_size=window_size)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed = 0

    args = getArguments()

    main(**vars(args))
