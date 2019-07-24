import argparse
import logging
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
import doc_similarity as ds


def getArguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', type=str, required=True, help='Format: <ID>\t<sentence>')
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-e', '--embeddings', type=str, required=True)
  parser.add_argument('-sl', '--stopword_language', type=str, default=None, help='The language code of the input else no stopword removal')

  return parser.parse_args()


def get_avg_doc_vec(x, w2v, stopwords):
    stopwords = ds.general_stopwords.union(stopwords)

    vec = [w2v[w] for w in x if w in w2v and w not in stopwords]

    if len(vec) == 0:
        return np.random.uniform(low=0.000001, high=0.000002, size=w2v.vector_size) # nearly zero vector

    return np.mean(vec, axis=0)


def main(input, output, embeddings, stopword_language=None):
    w2v = KeyedVectors.load_word2vec_format(embeddings, unicode_errors='replace', binary=False)

    stopwords_set = set()
    if stopword_language:
        stopwords_set = set(stopwords.words(ds.lang_map[stopword_language]))

    num_lines = 0
    with open(input, 'r') as fin:
        for _ in fin:
            num_lines += 1

    with open(input, 'r') as fin, open(output, 'w') as fout:
        fout.write('{} {}\n'.format(num_lines, w2v.vector_size))

        for line in tqdm(fin, total=num_lines):
            id, text = line.split('\t')
            vec = get_avg_doc_vec(text.split(), w2v, stopwords_set)
            fout.write('{} {}\n'.format(id, ' '.join(['{0:.6f}'.format(v) for v in vec])))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    np.random.seed = 0

    args = getArguments()
    main(**vars(args))
