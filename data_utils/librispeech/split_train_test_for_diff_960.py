import os
import numpy as np

if __name__ == '__main__':
    f = open('diff_meta_960.txt')
    train_f = open('train_diff_meta_960.txt', 'w')
    test_f = open('test_diff_meta_960.txt', 'w')
    a = [i.strip() for i in f]
    np.random.shuffle(a)
    len_all = len(a)
    len_train = int(len_all * 0.8)
    for i, x in enumerate(a):
        if i < len_train:
            train_f.write(x + '\n')
        else:
            test_f.write(x + '\n')
    