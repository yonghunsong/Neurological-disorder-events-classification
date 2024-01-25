import argparse
import multiprocessing as mp
import os
import time
import joblib
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from data_utils import (generate_evaluate_image_dataloaders,
                        generate_evaluate_time_dataloaders,
                        generate_evaluate_image_3CH_dataloaders)
from models import WavenetClassifier, resnet50, EfficientNetClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from efficientnet_pytorch import EfficientNet
import re


__MODEL__ = {
    'wavenet' : WavenetClassifier,
    'resnet50': resnet50,
    'efficientnet' : EfficientNet,
    }


def wavenet_evaluate(args, model, testloader, pretrained_id, model_path):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred, y_probability = [], [], []

    for idx, (data, labels) in enumerate(testloader):
        data = data.to(args.device, dtype=torch.float32)
        data = torch.unsqueeze(data, dim=0)
        
        with torch.no_grad():
            outputs = model.forward(data)

        _, predicted = torch.max(outputs, 1)
        
        labels =labels.to(args.device)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.squeeze()).sum().item()

        outputs_values = outputs.tolist()[0]
        y_probability.append(outputs_values)
        y_true.append(labels.squeeze().cpu().numpy().astype(int).item(0))
        y_pred.append(predicted.cpu().numpy().astype(int).item(0))
  
    test_accuracy = (correct_predictions / total_predictions) * 100.0
    
    return test_accuracy, y_true, y_pred, y_probability

def resnet50_evaluate(args, model, testloader, pretrained_id, model_path):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred, y_probability = [], [], []

    for idx, (data, labels)  in enumerate(testloader):

        temp_data_with_dimension  = data.unsqueeze(1)        
        new_data = temp_data_with_dimension .to(args.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model.forward(new_data)

        _, predicted = torch.max(outputs, 1)
        
        labels =labels.to(args.device)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.squeeze()).sum().item()

        outputs_values = outputs.tolist()[0]
        y_probability.append(outputs_values)
        y_true.append(labels.squeeze().cpu().numpy().astype(int).item(0))
        y_pred.append(predicted.cpu().numpy().astype(int).item(0))

    test_accuracy = (correct_predictions / total_predictions) * 100.0
    
    return test_accuracy, y_true, y_pred, y_probability

def efficientnet_evaluate(args, model, testloader, pretrained_id, model_path):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred, y_probability = [], [], []

    for idx, (data, labels)  in enumerate(testloader):

        new_data = data .to(args.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model.forward(new_data)

        _, predicted = torch.max(outputs, 1)
        
        labels =labels.to(args.device)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.squeeze()).sum().item()

        outputs_values = outputs.tolist()[0]
        y_probability.append(outputs_values)
        y_true.append(labels.squeeze().cpu().numpy().astype(int).item(0))
        y_pred.append(predicted.cpu().numpy().astype(int).item(0))

    test_accuracy = (correct_predictions / total_predictions) * 100.0
    
    return test_accuracy, y_true, y_pred, y_probability

def load_models(folder_path):
    model_path = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pth'):
            temp_model_path = os.path.join(folder_path, file_name)
            model_path.append(temp_model_path)
    return model_path

def calculate_accuracy(pred_list, true_list):
    correct = sum(1 for pred, true in zip(pred_list, true_list) if pred == true)
    total = len(pred_list)
    accuracy = correct / total * 100
    return accuracy

def evaluate(args, pretrained_model, input, dataloader, pretrained_id):

    folder_path = args.model_dir + pretrained_model + "_" + input
    model_path = load_models(folder_path)

    for cv, dataloader in enumerate(dataloader):
        
        y_true_list = []
        y_pred_list = []
        y_probability_list_temp = []
        y_test_array = []

        model_path.sort()
        for i in range(len(model_path)):

            if pretrained_model == 'wavenet':
                model = __MODEL__[getattr(args, f"model")](num_classes=args.num_classes,
                                window_size=args.window_size,
                                batch_size=args.batch_size).to(args.device)
            elif pretrained_model == 'resnet50':
                model = resnet50().to(args.device)
            elif pretrained_model == 'efficientnet':
                model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_classes).to(args.device)

            checkpoint = torch.load(model_path[i], map_location=torch.device('cpu'))
            model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(model_state_dict)

            model_name_extract = model_path[i].split('\\')[-1].split('.')[0]
            
            if pretrained_model == 'wavenet':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = wavenet_evaluate(args, model, dataloader, pretrained_id, model_name_extract)
            elif pretrained_model == 'resnet50':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = resnet50_evaluate(args, model, dataloader, pretrained_id, model_name_extract)
            elif pretrained_model == 'efficientnet':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = efficientnet_evaluate(args, model, dataloader, pretrained_id, model_name_extract)

            print(model_name_extract)
            print("test acc : {:.3f}".format(test_acc))
            y_test_array.append(test_acc)

            y_true_list = temp_y_true
            y_pred_list.append(temp_y_pred)
            y_probability_list_temp.append(temp_y_probability)

        return y_pred_list, y_true_list


def main(args):

    pretrained_id = time.time()

    save_dir = 'pretrainedEnsemble'
    os.makedirs(save_dir, exist_ok=True)
    pt_files = [file for file in os.listdir(save_dir) if file.endswith('.pt')]
    # Example
    # pt_files = ['gradient_boosting_2_4_5_6_result.pt', 'extra_trees_2_4_5_6_result.pt'] # 예시로 추가한 것이다.
    # pt_files = ['lightgbm_1_6_result.pt']

    dataset = 'Total'
    """
    (1) English, (2) French, (3) German, (4) Spanish, (5) Korean, (6)Total
    """

    count = 1

    dataloader_generator_1 = generate_evaluate_time_dataloaders(data_type = dataset, input = 'waveform')
    y_pred_list_1, y_true_list_1 = evaluate(args, "wavenet", "waveform", dataloader_generator_1, pretrained_id)
    dataloader_generator_2 = generate_evaluate_time_dataloaders(data_type = dataset, input = 'f0')
    y_pred_list_2, y_true_list_2 = evaluate(args, "wavenet", "f0", dataloader_generator_2, pretrained_id)
    dataloader_generator_3 = generate_evaluate_image_dataloaders(data_type = dataset, data_image = 'spectrogram')
    y_pred_list_3, y_true_list_3 = evaluate(args, "resnet50", "spectrogram", dataloader_generator_3, pretrained_id)
    dataloader_generator_4 = generate_evaluate_image_dataloaders(data_type = dataset, data_image = 'melspectrogram')
    y_pred_list_4, y_true_list_4 = evaluate(args, "resnet50", "melspectrogram", dataloader_generator_4, pretrained_id)
    dataloader_generator_5 = generate_evaluate_image_3CH_dataloaders(data_type = dataset, data_image = 'spectrogram')
    y_pred_list_5, y_true_list_5 = evaluate(args, "efficientnet", "spectrogram", dataloader_generator_5, pretrained_id)
    dataloader_generator_6 = generate_evaluate_image_3CH_dataloaders(data_type = dataset, data_image = 'melspectrogram')
    y_pred_list_6, y_true_list_6 = evaluate(args, "efficientnet", "melspectrogram", dataloader_generator_6, pretrained_id)

    for pt_file in pt_files:

        count = count + 1
        pretrained_models = re.findall(r'\d', pt_file)
        
        meta_input = []

        for pretrained_model in pretrained_models:

            if pretrained_model == '1': # wavenet_waveform
                y_true_list = y_true_list_1
                y_true_list = np.array(y_true_list)
                new_X_train_1 = np.array(y_pred_list_1)
                new_X_train_1 = new_X_train_1.tolist()
                y_true_list = y_true_list.tolist()

                meta_input.extend(new_X_train_1)
                
            elif pretrained_model == '2': # wavenet_f0
                y_true_list = y_true_list_2
                y_true_list = np.array(y_true_list)
                new_X_train_2 = np.array(y_pred_list_2)
                new_X_train_2 = new_X_train_2.tolist()
                y_true_list = y_true_list.tolist()
                meta_input.extend(new_X_train_2)

            elif pretrained_model == '3': # resnet50_spectrogram
                y_true_list = y_true_list_3
                y_true_list = np.array(y_true_list)
                new_X_train_3 = np.array(y_pred_list_3)
                new_X_train_3 = new_X_train_3.tolist()
                y_true_list = y_true_list.tolist()
                meta_input.extend(new_X_train_3)

            elif pretrained_model == '4': # resnet50_melspectrogram
                y_true_list = y_true_list_4
                y_true_list = np.array(y_true_list)
                new_X_train_4 = np.array(y_pred_list_4)
                new_X_train_4 = new_X_train_4.tolist()
                y_true_list = y_true_list.tolist()
                meta_input.extend(new_X_train_4)

            elif pretrained_model == '5': # efficientnet_spectrogram
                y_true_list = y_true_list_5
                y_true_list = np.array(y_true_list)
                new_X_train_5 = np.array(y_pred_list_5)
                new_X_train_5 = new_X_train_5.tolist()
                y_true_list = y_true_list.tolist()
                meta_input.extend(new_X_train_5)

            elif pretrained_model == '6': # efficientnet_melspectrogram
                y_true_list = y_true_list_6
                y_true_list = np.array(y_true_list)
                new_X_train_6 = np.array(y_pred_list_6)
                new_X_train_6 = new_X_train_6.tolist()
                y_true_list = y_true_list.tolist()
                meta_input.extend(new_X_train_6)



        meta_input = np.array(meta_input)
        meta_input = np.transpose(meta_input)
        ensemble_model_path = os.path.join(save_dir, pt_file)

        if 'gradient_boosting' in pt_file:
            ensemble_model = joblib.load(ensemble_model_path)
        elif 'random_forest' in pt_file:
            ensemble_model = joblib.load(ensemble_model_path)
        elif 'xgboost' in pt_file:
            ensemble_model = xgb.Booster()
            ensemble_model.load_model(ensemble_model_path)
        elif 'lightgbm' in pt_file:
            ensemble_model = lgb.Booster(model_file=ensemble_model_path)
        elif 'adaboost' in pt_file:
            ensemble_model = joblib.load(ensemble_model_path)
        elif 'extra_trees' in pt_file:
            ensemble_model = joblib.load(ensemble_model_path)
        elif 'svm' in pt_file:
            ensemble_model = joblib.load(ensemble_model_path)
            ensemble_model.probability = True

        predicted_labels = []
        if 'xgboost' in pt_file:
            dmatrix = xgb.DMatrix(np.array(meta_input))
            probas = ensemble_model.predict(dmatrix)
            max_prob_indices = np.argmax(probas, axis=1)
            predicted_labels.extend(max_prob_indices)

        elif 'lightgbm' in pt_file:
            probas = ensemble_model.predict(meta_input)
            max_prob_indices = np.argmax(probas, axis=1)
            predicted_labels.extend(max_prob_indices)

        elif 'svm' in pt_file:
            decision_values = ensemble_model.decision_function(meta_input)
            probas = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
            max_prob_indices = np.argmax(probas, axis=1)
            predicted_labels.extend(max_prob_indices)

        else:    
            predicted_labels = ensemble_model.predict(meta_input)
            probas = ensemble_model.predict_proba(meta_input)


        accuracy = accuracy_score(y_true_list, predicted_labels)
        accuracy = accuracy * 100
        print("Ensemble Accuracy: {:.3f}".format(accuracy))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Voice Event Detection")

    parser.add_argument('--dataset', type=str, default="English")
    """
    (1) English, (2) French, (3) German, (4) Spanish, (5) Korean, (6) Total
    """

    parser.add_argument('--model_dir', type=str, default="./Output/")
    parser.add_argument(f'--model', type=str, default="wavenet")
    """
    (1) wavenet, (2) resnet50, (3) efficientnet
    """
    parser.add_argument(f'--input', type=str, default="f0")
    """
    (1) waveform, (2) f0, (3) spectrogram, (4) melspectrogram
    """
    parser.add_argument('--save_dir', type=str, default="./Output_ensemble/")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=4000)

    args = parser.parse_args()

    mp.set_start_method('spawn')

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.num_classes = 4
    print(args)
    main(args)
