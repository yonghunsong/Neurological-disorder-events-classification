import argparse
import os
import torch
import numpy as np
import multiprocessing as mp
from ensemble_models import final_meta_model
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import itertools

def make_train_list(model):
    
    base_folder_path = "E:\\VED\\Output"
    folder_name = model
    folder_path = os.path.join(base_folder_path, folder_name)

    excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    excel_files.sort()

    predictions = []
    trues = []

    for file_name in excel_files:
        excel_path = os.path.join(folder_path, file_name)
        print(excel_path)

        df = pd.read_excel(excel_path)

        if "Prediction label" in df.columns:
            predictions.append(df["Prediction label"].values[:1600])
        else:
            raise ValueError("The column name 'Prediction label' does not exist in the Excel file: {}".format(file_name))

        if "True label" in df.columns:
            trues.append(df["True label"].values[:1600])
        else:
            raise ValueError("The column name 'True label' does not exist in the Excel file: {}".format(file_name))

    prediction = np.concatenate(predictions)
    true = np.concatenate(trues)

    return prediction, true


def train_models(args, r, model_index, combinations, save_dir):

    for combination in combinations:
        combination_names = [model_index[int(idx)-1] for idx in combination]

        predictions = []

        for i in combination_names:

            model_name = getattr(args, f'model_{i}')
            prediction, true = make_train_list(model_name)
            predictions.append(prediction)

        new_X_train = np.array(predictions)
        new_X_train = np.transpose(new_X_train)
        new_Y_train = np.array(true) # feature를 하나로 묶어서 학습시 이거사용, 즉 5-->1벡터
        new_Y_train = np.transpose(new_Y_train)

        indices = np.arange(new_X_train.shape[0])
        np.random.shuffle(indices)

        train_size = int(0.9 * len(indices))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_X = new_X_train[train_indices]
        train_Y = new_Y_train[train_indices]
        test_X = new_X_train[test_indices]
        test_Y = new_Y_train[test_indices]

        ensemble_model = args.ensemble_model
        model = final_meta_model(ensemble_model)
        model.fit(train_X, train_Y)

        os.makedirs(save_dir, exist_ok=True)
        result_model_name = f'{args.ensemble_model}_{"_".join(combination_names)}_result.pt'
        model_path = os.path.join(save_dir, result_model_name)

        if ensemble_model == 'gradient_boosting':
            joblib.dump(model, model_path)
        elif ensemble_model == 'random_forest':
            joblib.dump(model, model_path)
        elif ensemble_model == 'xgboost':
            model.save_model(model_path)
        elif ensemble_model == 'lightgbm':
            model.booster_.save_model(model_path)
        elif ensemble_model == 'catboost':
            model.save_model(model_path)
        elif ensemble_model == 'adaboost':
            joblib.dump(model, model_path)
        elif ensemble_model == 'extra_trees':
            joblib.dump(model, model_path)
        elif ensemble_model == 'svm':
            joblib.dump(model, model_path)
        elif ensemble_model == 'logistic_regression':
            joblib.dump(model, model_path)

        print(f"Model saved at {model_path}")

        prediction = model.predict(test_X)
        accuracy = accuracy_score(test_Y, prediction)
        print(f"Accuracy: {accuracy}")


def main(args):

    save_dir = 'pretrainedEnsemble'
    os.makedirs(save_dir, exist_ok=True)

    model_index = list(map(str, range(1, 7)))   

    for r in range(2, 7):
        combinations = list(itertools.combinations(model_index, r))
        train_models(args, r, model_index, combinations, save_dir)




if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Voice Event Detection")

    parser.add_argument('--file_dir', type=str, default="./Output/")

    parser.add_argument(f'--model_1', type=str, default="wavenet_waveform")
    parser.add_argument(f'--model_2', type=str, default="wavenet_f0")
    parser.add_argument(f'--model_3', type=str, default="resnet50_spectrogram")
    parser.add_argument(f'--model_4', type=str, default="resnet50_melspectrogram")
    parser.add_argument(f'--model_5', type=str, default="efficientnet_spectrogram")
    parser.add_argument(f'--model_6', type=str, default="efficientnet_melspectrogram")
    """
    model index number
    (1) wavenet, (2) wavenet_f0, (3) resnet50_spectrogram, (4) resnet50_melspectrogram, (5) efficientnet_spectrogram, (6) efficientnet_melspectrogram
    """
    """
    image processing method: (1) spectrogram, (2) melspectrogram
    """

    parser.add_argument('--ensemble_model', type=str, default="gradient_boosting")
    """
    (1) gradient_boosting, (2) random_forest, (3) xgboost, (4) lightgbm, (5) adaboost, (6) extra_trees, (7) svm
    """

    args = parser.parse_args()
    mp.set_start_method('spawn')

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.num_classes = 4
    
    print(args)
    
    main(args)
