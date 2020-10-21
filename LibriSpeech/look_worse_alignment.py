import os
import numpy as np

PPG_DIM = 347

def onehot(arr, depth, dtype=np.float32):
    assert len(arr.shape) == 1 #不为1则异常
    onehots = np.zeros(shape=[len(arr), depth], dtype=dtype)
    arr=arr.astype(np.int64)

    arr = arr-1  #下标从0开始
    arr = arr.tolist()

    onehots[np.arange(len(arr)), arr] = 1
    return onehots

def get_single_data_pair(fname, mfcc_dir, ppg_dir, f):
    assert os.path.isdir(mfcc_dir) and os.path.isdir(ppg_dir)

    mfcc_f = os.path.join(os.path.join(os.path.join(mfcc_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#fname+'.npy')
    ppg_f = os.path.join(os.path.join(os.path.join(ppg_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')

    mfcc = np.load(mfcc_f)
    ppg = np.load(ppg_f)
    ppg_raw = np.copy(ppg)
    ppg = onehot(ppg, depth=PPG_DIM)
    print('id', fname)
    print('mfcc', mfcc.shape[0], 'ppg-onehot', ppg.shape[0], 'ppg-raw', ppg_raw.shape[0])
    f.write(fname + '\n')
    f.write('mfcc ' + str(mfcc.shape[0]) + 'ppg-onehot' + str(ppg.shape[0]) + 'ppg-raw' + str(ppg_raw[0]) + '\n')
    
    return mfcc, ppg



if __name__ == '__main__':
    f_all = open('meta_960.txt')
    f_good = open('diff_meta_960.txt')
    f_worse = open('same_meta_960_detail.txt', 'w')
    a = [i.strip() for i in f_all]

    b = [i.strip() for i in f_good]
    good_dict = {}
    
    for x in b:
        good_dict[x] = 1

    for x in a:
        if good_dict.get(x, 0) == 0:
            tag = get_single_data_pair(x, './MFCCs', './PPGs', f_worse)
            # f_worse.write(x + '\n')