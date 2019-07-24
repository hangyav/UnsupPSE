import argparse
import logging
from tqdm import tqdm
import langid
import re
from itertools import zip_longest

import worker_consumer as wc
import doc_similarity as ds

simple_url_pattern = re.compile('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', re.IGNORECASE)


def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', type=str, required=True, help='Source language sentences')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target language sentences')
    parser.add_argument('-sc', '--scores', type=str, required=True, help='Initial scores')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-n', '--num_threads', default=1, type=int)
    parser.add_argument('-fl', '--from_language', type=str, default='de')
    parser.add_argument('-tl', '--to_language', type=str, default='en')
    parser.add_argument('--min_sent_length', default=3, type=int, help='Return 0.0 score if either src or trg sentence is shorter')
    parser.add_argument('--max_sent_len_diff', default=15, type=int, help='Return 0.0 score if src/trg sentence length difference is larger')
    parser.add_argument('--number_url_ratio_threshold', default=0.6, type=float, help='Return 0.0 score if either src or trg sentence has too much URLs')

    return parser.parse_args()


def get_lang_with_langid(text):
    return langid.classify(text)[0]


def helper(args, fr_stopwords, to_stopwords, min_sent_length=3,
           max_sent_len_diff=15, number_url_ratio_threshold=0.6,
           from_language='de', to_language='en', **kwargs):
    s, t, sc = args
    source = s.split()
    target = t.split()

    if sc <= 0.0:
        return sc

    source_len = len(source)
    target_len = len(target)

    if source_len < min_sent_length or target_len < min_sent_length:
        return 0.0

    if abs(source_len-target_len) >= max_sent_len_diff:
        return 0.0

    source_num_numbers_urls = len(ds.digits_pattern.findall(s)) + len(simple_url_pattern.findall(s))
    target_num_numbers_urls = len(ds.digits_pattern.findall(t)) + len(simple_url_pattern.findall(t))

    if float(source_num_numbers_urls)/float(source_len) > number_url_ratio_threshold \
            or float(target_num_numbers_urls)/float(target_len) > number_url_ratio_threshold:
        return 0.0

    if get_lang_with_langid(s) != from_language:
        return 0.0
    if get_lang_with_langid(t) != to_language:
        return 0.0

    return sc


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


def main(source, target, scores, output, from_language='de', to_language='en',
         num_threads=1, min_sent_length=3, max_sent_len_diff=15,
         number_url_ratio_threshold=0.6):

    langid.set_languages([from_language, to_language])

    fr_stopwords = set(ds.stopwords.words(ds.lang_map[from_language]))
    to_stopwords = set(ds.stopwords.words(ds.lang_map[to_language]))


    with open(output, 'w') as fout, tqdm() as pbar:
      wc.server(helper, writer, reader(source, target, scores), num_threads=num_threads, output_stream=fout, fr_stopwords=fr_stopwords,
                to_stopwords=to_stopwords, pbar=pbar, min_sent_length=min_sent_length, max_sent_len_diff=max_sent_len_diff,
                number_url_ratio_threshold=number_url_ratio_threshold, from_language=from_language, to_language=to_language)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = getArguments()

    main(**vars(args))
