import os
from tqdm import tqdm
import numpy as np

# in
good_path = 'same_meta_960_v2_cp.txt'
last_train_path = 'train_diff_meta_960.txt'
last_test_path = 'test_diff_meta_960.txt'

# out
train_path = 'train_same_meta_960_v2.txt'
test_path = 'test_same_meta_960_v2.txt'


def getList(f):
    a = open(f, 'r').readlines()
    a = [i.strip() for i in a]
    print(a[:3])
    return a


def main():
    good_file_dict = {}
    a = getList(good_path)
    for i in a:
        good_file_dict[i] = 1
    
    b = getList(last_train_path)
    fb = open(train_path, 'w')
    for i in b:
        if good_file_dict.get(i, 0) == 1:
            fb.write(i + '\n')

    c = getList(last_test_path)
    fc = open(test_path, 'w')
    for i in c:
        if good_file_dict.get(i, 0) == 1:
            fc.write(i + '\n')



if __name__ == '__main__':
    main()
