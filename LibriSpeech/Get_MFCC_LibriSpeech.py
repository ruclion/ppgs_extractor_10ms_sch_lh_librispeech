import argparse
import os
import numpy as np
from audio import wav2mfcc_v2, load_wav
from audio import hparams as audio_hparams

hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  #
    'min_db': -80.0,  # restrict the dynamic range of log power
    'iterations': 100,  # griffin_lim #iterations
    'silence_db': -28.0,
    'center': False,
}

assert hparams == audio_hparams

# wav_dir = 'wavs_16000_960'
wav_dir = '/ceph/dataset/LibriSpeech'
mfcc_dir = './MFCCs'

def main():
    #这一部分用于处理LibriSpeech格式的数据集。
    all_librispeech_flac = 0
    for first_dir in os.listdir(wav_dir): # train-clean-100
        if os.path.isfile(os.path.join(wav_dir, first_dir)):
            continue
        for second_dir in os.listdir(os.path.join(wav_dir, first_dir)): # 103
            for third_dir in os.listdir(os.path.join(os.path.join(wav_dir,first_dir), second_dir)): # 1240
                third_mfcc_dir = os.path.join(os.path.join(os.path.join(mfcc_dir,first_dir),second_dir), third_dir)
                third_mfcc_dir_without_first = os.path.join(os.path.join(mfcc_dir,second_dir), third_dir)
                third_wav_dir = os.path.join(os.path.join(os.path.join(wav_dir,first_dir),second_dir), third_dir)
                #print('Now in the '+mfcc_dir+' from '+ third_wav_dir)
                if not os.path.exists(third_mfcc_dir_without_first):
                    os.makedirs(third_mfcc_dir_without_first)

                wav_files = [os.path.join(third_wav_dir, f) for f in os.listdir(third_wav_dir) if f.endswith('.flac')]
                print('Extracting MFCC from {} to {}...'.format(third_wav_dir, third_mfcc_dir))
                cnt = 0
                # print('nnnnnnn')
                for wav_f in wav_files:
                    wav_arr = load_wav(wav_f, sr=hparams['sample_rate'])
                    mfcc_feats = wav2mfcc_v2(wav_arr, sr=hparams['sample_rate'],
                                            n_mfcc=hparams['n_mfcc'], n_fft=hparams['n_fft'],
                                            hop_len=hparams['hop_length'], win_len=hparams['win_length'],
                                            window=hparams['window'], num_mels=hparams['num_mels'],
                                            center=hparams['center'])
                    save_name = wav_f.split('/')[-1].split('.')[0] + '.npy'
                    save_name = os.path.join(third_mfcc_dir_without_first, save_name)
                    np.save(save_name, mfcc_feats)
                    cnt += 1
                    print(cnt)
                    all_librispeech_flac += 1
                    print('all cnt:', all_librispeech_flac)
                    # break
                # break
            # break
        # break
        # 修正: 提取完毕以后，需要手动将3个文件夹的东西mv到同一个，和ppg一样的2338个文件夹 -> third_mfcc_dir_without_first使用它就不需要mv了
    return


if __name__ == '__main__':
    main()
