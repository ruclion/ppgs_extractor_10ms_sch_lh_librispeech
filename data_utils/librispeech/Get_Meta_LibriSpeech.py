import os
from tqdm import tqdm
import numpy as np


mfccs_dir = 'MFCCs'
ppgs_dir = 'PPGs'
meta_list_fromPPGs = []
meta_list_fromMFCCs = []

meta_list_final = []
meta_list = []

meta_path = 'meta_960.txt'
train_path = 'train_960.txt'
test_path = 'test_960.txt'

mfcc_dict = {}

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

    # 同时存在的才要，alignment很珍贵，不过应当完全重合，所以就不写no到文件中了，print看看就行
    # 但是结果是：PPGs: 281235，MFCCs: 281241；Final Lists: 281235，所以ppg是mfcc的子集，还好

    for x in tqdm(meta_list_fromPPGs):
        if mfcc_dict.get(x, 0) == 1:
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
