import argparse
import cPickle
import sys


def main(vocab_file, token_map, vocab):
    with open(vocab_file) as r:
        for i, l in enumerate(r):
            word = l.strip()
            if word in token_map:
                word = token_map[word]
            vocab[word] = i + 2
            sys.stderr.write(word + "\n")
    output = "{}.pkl".format(vocab_file)
    cPickle.dump(vocab, open(output, "w"), cPickle.HIGHEST_PROTOCOL)
    print "--------------"
    print "dumped to {}".format(output)


if __name__ == '__main__':
    token_map = {"<unk>": "UNK", "<s>": "<s>", "</s>": "</s>"}
    vocab = {"UNK": 1, "<s>": 0, "</s>": 0}
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_in_text", type=str)
    args = parser.parse_args()
    main(args.vocab_in_text, token_map, vocab)
