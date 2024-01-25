import os
import torch
from data_preprocessor import TrainDataset, VoiceDataset
from augment import augment_data
import numpy as np
from data_preprocessor import normalize_matrix
from image_processing import Mel_Spectrogram, Spectrogram, Spectrogram_3CH, Mel_Spectrogram_3CH
import pysptk


__DATA_PATH__ = {
    'five_fold'    : "./Data/train/five_fold/",
    'postech_eng' : "./Data/train/English",
    'cefir_eng' : "./Data/test/English",
    'cefir_fr' : "./Data/test/French",
    'cefir_germ' : "./Data/test/Germany",
    'cefir_kor' : "./Data/test/Korean",
    'cefir_spa' : "./Data/test/Spanish"
}

__DATASET__ = {
    'five_fold'   : TrainDataset,
    'postech_eng' : TrainDataset,
    'cefir_eng' : VoiceDataset,
    'cefir_fr' : VoiceDataset,
    'cefir_germ' : VoiceDataset,
    'cefir_kor' : VoiceDataset,
    'cefir_spa' : VoiceDataset
}


def load_dataset(fold_idx, data_path):
    trainset = {'data': [], 'labels': []}
    validset = {'data': [], 'labels': []}

    for idx in range(5):
        data = np.load(os.path.join(data_path, f"fold{idx}_data.npy"), allow_pickle=True).item()
        if idx == fold_idx:
            validset['data'].extend(data['data'])
            validset['labels'].extend(data['labels'])
        else:
            trainset['data'].extend(data['data'])
            trainset['labels'].extend(data['labels'])

    return np.array(trainset['data']), np.array(trainset['labels']), np.array(validset['data']), np.array(validset['labels'])


def generate_fold_to_cv_dataloaders(data_type='five_fold', num_fold=5, batch_size=16, window_size=1000, slide_size=500):
    
    data_path = "./Data/train/five_fold/"

    for fold_idx in range(5):
        trainset, train_labels, validset, valid_labels = load_dataset(fold_idx, data_path)

        augmented_trainset = []
        testset = []

        for idx in range(len(trainset)):
            train_data = trainset[idx]
            train_label = train_labels[idx]

            augmented_trainset.append((train_data, train_label))
            augmented_data = augment_data(train_data, n_augment=3, flip_prob=0.5, shift_max=10, scale_min=0.9, scale_max=1.1, noise_level=0.1)
            for augmented_sample in augmented_data:
                augmented_trainset.append((augmented_sample, train_label))

        for idx in range(len(validset)):
            test_data = validset[idx]
            test_label = valid_labels[idx]
            testset.append((test_data, test_label))

        trainLoader = torch.utils.data.DataLoader(dataset=augmented_trainset, batch_size=batch_size, num_workers=1, shuffle=True)
        validLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=1)

        yield {'train' : trainLoader, 'valid' : validLoader}

def generate_f0_fold_to_cv_dataloaders(data_type='five_fold', num_fold=5, batch_size=16, window_size=1000, slide_size=500):
    
    data_path = "./Data/train/five_fold/"

    for fold_idx in range(5):
        trainset, train_labels, validset, valid_labels = load_dataset(fold_idx, data_path)

        augmented_trainset = []
        testset = []

        for idx in range(len(trainset)):
            train_data = trainset[idx]
            train_label = train_labels[idx]
            f0 = f0_data(train_data)         

            for augmented_sample in f0:
                augmented_trainset.append((augmented_sample, train_label))

        for idx in range(len(validset)):
            test_data = validset[idx]
            test_label = valid_labels[idx]

            test_f0 = f0_data(test_data)

            for augmented_sample in test_f0:
                testset.append((augmented_sample, test_label))

        trainLoader = torch.utils.data.DataLoader(dataset=augmented_trainset, batch_size=batch_size, num_workers=1, shuffle=True)
        validLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=1)

        yield {'train' : trainLoader, 'valid' : validLoader}

def f0_data(data):

    norm_sig = (data - np.min(data)) / (np.max(data) - np.min(data))
    norm_sig = norm_sig * 50000 - 25000
    f0_data_value = pysptk.rapt(norm_sig.astype(np.float32), fs=6400, hopsize=1, min=10, max=500)
    f0_data_value = np.array(f0_data_value)

    return f0_data_value

def generate_fold_to_cv_image_dataloaders(data_type='five_fold', num_fold=5, batch_size=64, slide_size=500):
    
    data_path = "./Data/train/five_fold/"

    for fold_idx in range(5):
        trainset, train_labels, validset, valid_labels = load_dataset(fold_idx, data_path)

        augmented_trainset = []
        testset = []

        for idx in range(len(trainset)):
            train_data = trainset[idx]
            train_label = train_labels[idx]
            train_iamge_sample = Mel_Spectrogram(train_data)
            augmented_trainset.append((normalize_matrix(train_iamge_sample), train_label))

            augmented_data = augment_data(train_data, n_augment=3, flip_prob=0.5, shift_max=10, scale_min=0.9, scale_max=1.1, noise_level=0.1)

            for augmented_sample in augmented_data:
                augmented_iamge_sample = Mel_Spectrogram(augmented_sample)
                augmented_trainset.append((normalize_matrix(augmented_iamge_sample), train_label))

        for idx in range(len(validset)):
            test_data = validset[idx]
            test_label = valid_labels[idx]

            validation_iamge_sample = Mel_Spectrogram(test_data)

            testset.append((normalize_matrix(validation_iamge_sample), test_label))

        trainLoader = torch.utils.data.DataLoader(dataset=augmented_trainset, batch_size=batch_size, shuffle=True)
        validLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

        yield {'train' : trainLoader, 'valid' : validLoader}


def load_test_dataset(data_path):
    testset = {'data': [], 'labels': []}
    folder_path = data_path
    file_names = sorted(os.listdir(folder_path))
    for file_name in file_names:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            testset['data'].extend(data)
            if 'cough' in file_name:
                testset['labels'].extend([0] * len(data))
            elif 'speak' in file_name:
                testset['labels'].extend([1] * len(data))
            elif 'swallow' in file_name:
                testset['labels'].extend([2] * len(data))
            elif 'throatclear' in file_name:
                testset['labels'].extend([3] * len(data))
    
    return np.array(testset['data']), np.array(testset['labels'])

def generate_evaluate_time_dataloaders(data_type="English", input='waveform'):
    
    data_path = "./Data/test/" + data_type + "/"

    test_set, test_labels = load_test_dataset(data_path)
    testset = []

    for idx in range(len(test_set)):
        test_data = test_set[idx]
        test_label = test_labels[idx]

        if input == 'f0':
            f0 = f0_data(test_data)
            test_data = f0

        testset.append((test_data, test_label))

    testLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)
    yield testLoader


def generate_evaluate_image_dataloaders(data_type='English', data_image='spectrogram'):
    
    data_path = "./Data/test/" + data_type + "/"

    test_set, test_labels = load_test_dataset(data_path)
    testset = []

    if data_image == 'spectrogram':
        image_type = Spectrogram
    elif data_image == 'melspectrogram':
        image_type = Mel_Spectrogram

    for idx in range(len(test_set)):
        test_data = test_set[idx]
        test_iamge_sample = image_type(test_data)
        test_label = test_labels[idx]

        testset.append((normalize_matrix(test_iamge_sample), test_label))

    testLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)
    yield testLoader

def generate_evaluate_image_3CH_dataloaders(data_type='English', data_image='spctrogram'):
    
    data_path = "./Data/test/" + data_type + "/"

    test_set, test_labels = load_test_dataset(data_path)
    testset = []

    if data_image == 'spectrogram':
        image_type = Spectrogram_3CH
    elif data_image == 'melspectrogram':
        image_type = Mel_Spectrogram_3CH

    for idx in range(len(test_set)):
        test_data = test_set[idx]
        test_iamge_sample = image_type(test_data)
        test_label = test_labels[idx]
        
        testset.append((normalize_matrix(test_iamge_sample), test_label))

    testLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)
    yield testLoader
