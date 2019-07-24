import argparse
import logging


def getArguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-p', '--prediction', type=str, required=True, help='Format: <SRC_ID>\t<TRG_ID>')
  parser.add_argument('-g', '--gold', type=str, required=True, help='Format: <SRC_ID>\t<TRG_ID>')

  return parser.parse_args()


def main(prediction, gold):
    gold_labels = dict()
    with open(gold, 'r') as fin:
        for line in fin:
            fr, en = line.split('\t')
            if fr in gold_labels:
                raise ValueError('Found ID multiple times in gold: {}'.format(fr))

            gold_labels[fr] = en

    tp = 0
    fp = 0
    N = len(gold_labels)
    seen = set()
    with open(prediction, 'r') as fin:
        for line in fin:
            fr, en = line.split('\t')
            if fr in seen:
                raise ValueError('Found ID multiple times in prediction: {}'.format(fr))

            seen.add(fr)
            if fr in gold_labels:
                if en == gold_labels[fr]:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1

    fn = N - tp
    logger.debug('tp:{} fp:{} fn:{} N:{}'.format(tp, fp, fn, N))

    P = 0.0
    R = 0.0
    F1 = 0.0
    if tp+fp > 0:
        P = tp/(tp+fp)
    if tp+fn > 0:
        R = tp/(tp+fn)
    if P+R > 0:
        F1 = (2*P*R)/(P+R)

    print('PRECISION,RECALL,F1')
    print('{},{},{}'.format(P,R,F1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = getArguments()

    main(**vars(args))
