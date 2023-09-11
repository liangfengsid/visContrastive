import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

def extract_feature(a :np.ndarray, window_size=100):
    features = []
    i = 0
    while i < a.shape[0]:
        window = a[i : i + window_size]
        features.append([np.max(window),np.min(window), np.mean(window), np.median(window), np.std(window)])
        i += window_size
    return np.array(features)

def process_split_data(preprocess_dir, files, key_prefix, sessions, sectors, test_ratio, channels, feature_size, window):
    
    time_train = np.empty([0, channels * feature_size])
    time_test = np.empty([0, channels * feature_size])
    emo_labels_train = np.empty([0, ], dtype=np.int32)
    emo_labels_test = np.empty([0, ], dtype=np.int32)
    subject_labels_train = np.empty([0, ], dtype=np.int32)
    subject_labels_test = np.empty([0, ], dtype=np.int32)
    labels = scipy.io.loadmat(preprocess_dir + '/label.mat')['label'][0]

    mod = (int)(1 / test_ratio)
    total_files = files.shape[0]

    for f in tqdm(range(total_files), desc="Processing files"):
        eegs = scipy.io.loadmat(f'{preprocess_dir}/{files[f]}')
        # per sector
        for i in range(1, sectors + 1):
            k = f'{key_prefix[f]}_eeg{str(i)}'
            epoch = np.apply_along_axis(extract_feature, 1, eegs[k], window).swapaxes(0, 1)
            epoch = epoch.reshape(epoch.shape[0], -1)
            label = np.full((epoch.shape[0], ), labels[i - 1], dtype=np.int32)
            subject_label = np.full((epoch.shape[0], ), f // sessions, dtype=np.int32)
            index = files.shape[0] * sectors + i
            if index % mod == 0:
                time_test = np.append(time_test, epoch, axis=0)
                emo_labels_test = np.append(emo_labels_test, label)
                subject_labels_test = np.append(subject_labels_test, subject_label)
            else:
                time_train = np.append(time_train, epoch, axis=0)
                emo_labels_train = np.append(emo_labels_train, label)
                subject_labels_train = np.append(subject_labels_train, subject_label)

    emo_labels_train = emo_labels_train + 1
    emo_labels_test = emo_labels_test + 1
    #print(time_train.shape, emo_labels_train.shape, subject_labels_train.shape)

    return time_train, time_test, emo_labels_train, emo_labels_test, subject_labels_train, subject_labels_test


def save_data(filename, time_train, time_test, emo_labels_train, emo_labels_test, subject_labels_train, subject_labels_test):
    with open(filename, 'wb') as f:
        np.save(f, time_train)
        np.save(f, time_test)
        np.save(f, emo_labels_train)
        np.save(f, emo_labels_test)
        np.save(f, subject_labels_train)
        np.save(f, subject_labels_test)


def load_split_data(split_path):
    with open(split_path, 'rb') as f:
        time_train = np.load(f)
        time_test = np.load(f)
        emo_labels_train = np.load(f)
        emo_labels_test = np.load(f)
        subject_labels_train = np.load(f)
        subject_labels_test = np.load(f)
    
    return time_train, time_test, emo_labels_train, emo_labels_test, subject_labels_train, subject_labels_test

def main(config):
    preprocess_dir = config['preprocess_dir']
    channels = config['channels']
    feature_size = config['feature_size']
    persons = config['persons']
    sessions = config['sessions']
    sectors = config['sectors']
    key_prefix = config['key_prefix']
    save_path = config['save_path']
    test_ratio = config['test_ratio']
    windows = config['windows']

    files = os.listdir(preprocess_dir)
    files.sort()
    files = np.asarray(files)[: sessions * persons]

    for window in windows:
        time_train, time_test, emo_labels_train, emo_labels_test, subject_labels_train, subject_labels_test = \
            process_split_data(preprocess_dir, files, key_prefix, sessions, sectors, test_ratio, channels, feature_size, window)

        save_data(save_path+'/time_reduced_w{window}.npy', time_train, time_test, emo_labels_train, emo_labels_test, 
                                    subject_labels_train, subject_labels_test)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
