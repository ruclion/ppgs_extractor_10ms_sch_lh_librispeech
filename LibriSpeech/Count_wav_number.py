import argparse
import os
import numpy as np
# from audio import wav2mfcc_v2, load_wav
# from audio import hparams as audio_hparams

# hparams = {
#     'sample_rate': 16000,
#     'preemphasis': 0.97,
#     'n_fft': 400,
#     'hop_length': 160,
#     'win_length': 400,
#     'num_mels': 80,
#     'n_mfcc': 13,
#     'window': 'hann',
#     'fmin': 30.,
#     'fmax': 7600.,
#     'ref_db': 20,  #
#     'min_db': -80.0,  # restrict the dynamic range of log power
#     'iterations': 100,  # griffin_lim #iterations
#     'silence_db': -28.0,
#     'center': False,
# }

# assert hparams == audio_hparams

# wav_dir = 'wavs_16000_960'
wav_dir = '/ceph/dataset/LibriSpeech'
# mfcc_dir = './MFCCs'

def main():
    #这一部分用于处理LibriSpeech格式的数据集。
    all_librispeech_flac = 0
    all_librispeech_canLoad = 0
    for first_dir in os.listdir(wav_dir): # train-clean-100
        if os.path.isfile(os.path.join(wav_dir, first_dir)):
            continue
        print(first_dir)
        for second_dir in os.listdir(os.path.join(wav_dir, first_dir)): # 103
            for third_dir in os.listdir(os.path.join(os.path.join(wav_dir,first_dir), second_dir)): # 1240
                third_wav_dir = os.path.join(os.path.join(os.path.join(wav_dir,first_dir),second_dir), third_dir)

                # wav_files = [os.path.join(third_wav_dir, f) for f in os.listdir(third_wav_dir) if f.endswith('.flac')]
                wav_files = [os.path.join(third_wav_dir, f) for f in os.listdir(third_wav_dir)]
                all_librispeech_flac += len(wav_files)
                # print('Extracting MFCC from {} to {}...'.format(third_wav_dir, third_mfcc_dir))
                # cnt = 0
                # print('nnnnnnn')
                # for wav_f in wav_files:
                    # wav_arr = load_wav(wav_f, sr=hparams['sample_rate'])
                    # all_librispeech_canLoad += 1
                    
                    # break
                # break
            # break
        # break
    print('all cnt:', all_librispeech_flac)
    print('all load cnt:', all_librispeech_canLoad)
    return


if __name__ == '__main__':
    main()
