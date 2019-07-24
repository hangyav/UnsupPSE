import argparse
import logging
from tqdm import tqdm
import re
import Levenshtein
import itertools
from collections import Counter
general_stopwords = {'&quot;', '&amp;', '&apos;', '&gt;', '&lt;', '&quot;'}

digits_pattern = re.compile('[0-9]+([,][0-9]+)?([.][0-9]+)?')


def getArguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--source', type=str, required=True, help='File containing source language corpus')
  parser.add_argument('-t', '--target', type=str, required=True, help='File containing target language corpus')
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-mal', '--max_length', default=-1, type=int, help='Maximum word length for filtering too long words')
  parser.add_argument('-mil', '--min_length', default=3, type=int, help='Minimum word length')
  parser.add_argument('-mif', '--min_frequency', default=5, type=int, help='Minimum word frequency; rare words are ignored')
  parser.add_argument('-th', '--threshold', default=0.8, type=float, help='Threshold for word similarity. Less similar word pairs are ignored')
  parser.add_argument('-k', '--k', default=2, type=int, help='Number of allowed insertion/deletion operations for generating word pair candidates. See: Wolf Garbe. 2012. 1000x faster spelling correction algorithm. http://blog.faroo.com/2012/06/07/improved-edit-distance- based-spelling-correction/')

  return parser.parse_args()


def get_vocab(input_file, remove_digits=True, min_length=4, max_length=-1, min_freq=5):
    logging.info('Reading vocabulary from: {}'.format(input_file))
    c = Counter()
    with open(input_file, 'r') as fin:
      for line in tqdm(fin):
        for w in line.split():
          len_w = len(w)
          if len_w < min_length or (0 < max_length < len_w):
            continue
          if remove_digits and digits_pattern.search(w) is not None:
            continue

          c.update([w])

    res = {w for w, f in c.items() if f >= min_freq}
    logging.info('Vocabulary size: {}'.format(len(res)))
    return res


## FROM Parker Riley, Daniel Gildea. 2018. Orthographic Features for Bilingual Lexicon Induction: https://github.com/Luminite2/vecmap/blob/master/ortho.py #####################################
def lexDeleteAugment(lex, k):
  d = {}
  for w in lex:
    #generate all types
    edits = allDeletesUpToK(w,k)
    #hash them all to w (add to list)
    for edit in edits:
      if edit not in d:
        d[edit] = [w]
      else:
        d[edit].append(w)
  return d


def matches(tmap, w, k):
  cands = []
  for d in allDeletesUpToK(w,k):
    if d in tmap:
      cands += tmap[d]
  return cands


def allDeletesUpToK(word, k):
  l = []
  for i in range(k+1):
    for poss in itertools.combinations(range(len(word)),i):
      w = word
      j = 0
      for p in poss:
        w = w[:p-j] + w[p-j+1:]
        j += 1
      l.append(w)
  return l

###############################################################################

def main(source, target, output, num_threads=1, threshold=0.6, min_length=3, max_length=-1, min_frequency=5, k=5):
    src = get_vocab(source, max_length=max_length, min_length=min_length, min_freq=min_frequency)
    tgt = get_vocab(target, max_length=max_length, min_length=min_length, min_freq=min_frequency)

    trgmap = lexDeleteAugment(tgt,k)

    with open(output, 'w') as fout:
      for s in tqdm(src):
        sims = [(t, Levenshtein.ratio(s, t)) for t in matches(trgmap, s, k)]
        sims = [item for item in sims if item[1] >= threshold]
        if len(sims) == 0:
            continue
        fout.write('{}\t{}\n'.format(s, '\t'.join(['{}\t{:10.4f}'.format(w, sim) for w, sim in sims])))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = getArguments()

    main(**vars(args))
