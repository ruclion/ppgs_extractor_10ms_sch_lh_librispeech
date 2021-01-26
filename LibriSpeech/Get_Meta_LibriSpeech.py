import os
from tqdm import tqdm
import numpy as np


mfccs_dir = 'MFCCs'
ppgs_dir = 'PPGs'
meta_list_fromPPGs = []
meta_list_fromMFCCs = []

meta_list_final = []

meta_path = 'meta_librispeech.txt'
train_path = 'train_meta_librispeech.txt'
test_path = 'test_meta_librispeech.txt'

mfcc_dict = {}

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
            print(fname, '---', mfcc.shape[0], ppg.shape[0])
            return False
    except:
        return False
    return mfcc, ppg


def main():
    # ppg
    for second_dir in os.listdir(ppgs_dir):
        for third_dir in os.listdir(os.path.join(ppgs_dir,second_dir)):
            third_wavs_dir = os.path.join(os.path.join(ppgs_dir,second_dir),third_dir)
            ppg_files = [f[:-4] for f in os.listdir(third_wavs_dir) if f.endswith('.npy')]
            # print('Extracting MFCC from {}...'.format(third_wavs_dir))
            meta_list_fromPPGs.extend(ppg_files)
            # break
        # break
    print('PPGs:', len(meta_list_fromPPGs))

    # mfcc
    for second_dir in os.listdir(mfccs_dir):
        for third_dir in os.listdir(os.path.join(mfccs_dir,second_dir)):
            third_wavs_dir = os.path.join(os.path.join(mfccs_dir,second_dir),third_dir)
            mfcc_files = [f[:-4] for f in os.listdir(third_wavs_dir) if f.endswith('.npy')]
            # print('Extracting MFCC from {}...'.format(third_wavs_dir))
            meta_list_fromMFCCs.extend(mfcc_files)
            for x in mfcc_files:
                mfcc_dict[x] = 1
            # break
        # break
    print('MFCCs:', len(meta_list_fromMFCCs))

    # PPG和MFCC均有的才放在final中
    for x in tqdm(meta_list_fromPPGs):
        if mfcc_dict.get(x, 0) == 1:
            if get_single_data_pair(x, mfccs_dir, ppgs_dir) is False:
                print('have but dim not same', x)
                continue
            meta_list_final.append(x)
        else:
            print('no:', x)
    print('Final Lists:', len(meta_list_final))


    # 写入文件
    f = open(meta_path, 'w')
    for idx in meta_list_final:
        f.write(idx + '\n')
    
    np.random.shuffle(meta_list_final)
    len_all = len(meta_list_final)
    len_train = int(len_all * 0.8)
    f_train = open(train_path, 'w')
    f_test = open(test_path, 'w')
    for i, x in enumerate(meta_list_final):
        if i < len_train:
            f_train.write(x + '\n')
        else:
            f_test.write(x + '\n')

    return


if __name__ == '__main__':
    main()
