#coding=utf-8
"""
My version of CNN Sentence classification Model
@author: cer
@forked_from: Yoon Kim
"""

from cer_model import CNN_Sen_Model
import json
import numpy as np
import cPickle


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)
    # for item in train:
    #     print len(item)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]


if __name__ == '__main__':
    with open("model.json", "r") as f:
        conf = json.load(f)
    # 加载模型数据
    print "loading data...",
    x = cPickle.load(open("mr.p", "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

    # 获取模型参数
    if conf['non_static']:
        print "model architecture: CNN-non-static"
    else:
        print "model architecture: CNN-static"
    if conf['word_vectors'] == "rand":
        print "using: random vectors"
        U = W2
    elif conf['word_vectors'] == "word2vec":
        print "using: word2vec vectors"
        U = W

    r = range(0, 1)
    for i in r:
        # print conf["max_l"], type(conf["max_l"])
        # print max(conf["filter_hs"]), type(max(conf["filter_hs"]))
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=conf["max_l"], k=300, filter_h=max(conf["filter_hs"]))
        model = CNN_Sen_Model(conf)
        model.build_model(U)
        perf = model.train(datasets)
        print "cv: " + str(i) + ", perf: " + str(perf)
