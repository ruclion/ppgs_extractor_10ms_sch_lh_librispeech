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

def get_single_data_pair(fname, mfcc_dir, ppg_dir):
    assert os.path.isdir(mfcc_dir) and os.path.isdir(ppg_dir)

    mfcc_f = os.path.join(os.path.join(os.path.join(mfcc_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#fname+'.npy')
    ppg_f = os.path.join(os.path.join(os.path.join(ppg_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')

    try:
        mfcc = np.load(mfcc_f)
        ppg = np.load(ppg_f)
        ppg = onehot(ppg, depth=PPG_DIM)
        if mfcc.shape[0] != ppg.shape[0]:
            # print(fname, '---', mfcc.shape[0], ppg.shape[0])
            return False
    except:
        return False
    return mfcc, ppg



if __name__ == '__main__':
    f = open('meta_960.txt')
    
    a = [i.strip() for i in f]
    dif_list = []
    out_list = []
    for i, x in enumerate(a): 
        tag = get_single_data_pair(x, './MFCCs', './PPGs')
        if tag is False:
            dif_list.append(x)
            # fdif.write(x + '\n')
        else:
            # print(x)
            out_list.append(x)
            # fout.write(x + '\n')
        if i % 1000 == 0:
            print('----:', i)

    print('check over, start write:')
    fout = open('same_meta_960_v2.txt', 'w')
    fdif = open('diff_meta_960_v2.txt', 'w')
    for x in out_list:
        fout.write(x + '\n')
    for x in dif_list:
        fdif.write(x + '\n')