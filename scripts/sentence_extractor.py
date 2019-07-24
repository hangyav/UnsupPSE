import argparse
import logging


def getArguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', type=str, required=True, help='Input sentences with IDs')
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-ids', '--ids', type=str, required=True, help='File containing IDs to extract')
  parser.add_argument('-col', '--column', default=0, type=int, help='Column to use from the ids file')

  return parser.parse_args()


def main(input, output, ids, column):
    with open(input, 'r') as fin:
        sentences = dict()
        for line in fin:
            idx, sent = line.strip().split('\t')
            sentences[idx] = sent

    with open(ids, 'r') as fin, open(output, 'w') as fout:
        for line in fin:
            idx = line.split('\t')[column].strip()
            if idx in sentences:
                fout.write('{}\n'.format(sentences[idx]))
            else:
                logging.warning('Sentence id does not exists. Skipping: {}'.format(idx))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = getArguments()

    main(**vars(args))
