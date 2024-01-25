import argparse
import multiprocessing as mp
import os
import time
import torch
from data_utils import (generate_evaluate_image_dataloaders,
                        generate_evaluate_time_dataloaders,
                        generate_evaluate_image_3CH_dataloaders)
from evaluation_indicator import (result_Confusion_ROC_AUC_save,
                                  result_Confusion_ROC_AUC_save_each_model,
                                  result_Confusion_ROC_AUC_save_one_model,
                                  result_Confusion_ROC_AUC_save_ensemble_model)

from models import WavenetClassifier, resnet50, EfficientNetClassifier
import numpy as np
from efficientnet_pytorch import EfficientNet

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

    result_Confusion_ROC_AUC_save_each_model(args, args.input, y_probability, y_true, args.num_classes, pretrained_id, model_path)
        
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

    result_Confusion_ROC_AUC_save_each_model(args, args.input, y_probability, y_true, args.num_classes, pretrained_id, model_path)
    
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

    result_Confusion_ROC_AUC_save_each_model(args, args.input, y_probability, y_true, args.num_classes, pretrained_id, model_path)

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

def evaluate(args, dataloader, pretrained_id):

    folder_path = args.model_dir + args.model + "_" + args.input
    model_path = load_models(folder_path)

    for cv, dataloader in enumerate(dataloader):
        
        y_true_list = []
        y_pred_list = []
        y_probability_list_temp = []
        y_test_array = []

        for i in range(len(model_path)):

            if args.model == 'wavenet':
                model = __MODEL__[getattr(args, f"model")](num_classes=args.num_classes,
                                window_size=args.window_size,
                                batch_size=args.batch_size).to(args.device)
            elif args.model == 'resnet50':
                model = resnet50().to(args.device)
            elif args.model == 'efficientnet':
                model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_classes).to(args.device)
            elif args.model == 'efficientnet_b7':
                model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.num_classes).to(args.device)

            checkpoint = torch.load(model_path[i], map_location=torch.device('cpu'))
            model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(model_state_dict)

            model_name_extract = model_path[i].split('\\')[-1].split('.')[0]
            
            if args.model == 'wavenet':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = wavenet_evaluate(args, model, dataloader, pretrained_id, model_name_extract)
            elif args.model == 'resnet50':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = resnet50_evaluate(args, model, dataloader, pretrained_id, model_name_extract)
            elif args.model == 'efficientnet':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = efficientnet_evaluate(args, model, dataloader, pretrained_id, model_name_extract)
            elif args.model == 'efficientnet_b7':
                test_acc, temp_y_true, temp_y_pred, temp_y_probability = efficientnet_evaluate(args, model, dataloader, pretrained_id, model_name_extract)

            print(model_name_extract)
            print("test acc : {:.3f}".format(test_acc))
            y_test_array.append(test_acc)

            y_true_list = temp_y_true
            y_pred_list.append(temp_y_pred)
            y_probability_list_temp.append(temp_y_probability) 

        y_pred_list_avg = np.mean(y_probability_list_temp, axis=0)
        max_indices = np.argmax(y_pred_list_avg, axis=1)
        y_pred_list_avg_result = max_indices.reshape(-1, 1)
        y_probability_list = [item for sublist in y_pred_list_avg_result for item in sublist]


        if args.model == 'wavenet':
            result_Confusion_ROC_AUC_save_one_model(args, 'wavenet', args.input, y_pred_list_avg, y_true_list, args.num_classes, pretrained_id, 'none')
        elif args.model == 'resnet50':
            result_Confusion_ROC_AUC_save_one_model(args, 'resnet50', args.input, y_pred_list_avg, y_true_list, args.num_classes, pretrained_id, 'none')
        elif args.model == 'efficientnet':
            result_Confusion_ROC_AUC_save_one_model(args, 'efficientnet', args.input, y_pred_list_avg, y_true_list, args.num_classes, pretrained_id, 'none')
        elif args.model == 'efficientnet_b7':
            result_Confusion_ROC_AUC_save_one_model(args, 'efficientnet_b7', args.input, y_pred_list_avg, y_true_list, args.num_classes, pretrained_id, 'none')

        accuracy = calculate_accuracy(y_probability_list, y_true_list)
        print("Probability Average Accuracy: {:.3f}".format(accuracy))
        y_test_array_average = sum(y_test_array) / len(y_test_array)
        print("Each_accuracy --> average: {:.3f}".format(y_test_array_average))


def main(args):

    pretrained_id = time.time()
    print("load data : " + args.dataset)
    print("load model : " + args.model)

    if args.model == 'wavenet':

        if args.input == 'waveform':
            print("waveform")
            dataloader_generator = generate_evaluate_time_dataloaders(data_type=args.dataset, input=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        elif args.input == 'f0':
            print("f0")
            dataloader_generator = generate_evaluate_time_dataloaders(data_type=args.dataset, input=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        else:
            print("wrong")

    elif args.model == 'resnet50':

        if args.input == 'spectrogram':
            print("spectrogram")
            dataloader_generator = generate_evaluate_image_dataloaders(data_type=args.dataset, data_image=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        elif args.input == 'melspectrogram':
            print("melspectrogram")
            dataloader_generator = generate_evaluate_image_dataloaders(data_type=args.dataset, data_image=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        else:
            print("wrong")

    elif args.model == 'efficientnet':

        if args.input == 'spectrogram':
            print("spectrogram")
            dataloader_generator = generate_evaluate_image_3CH_dataloaders(data_type=args.dataset, data_image=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        elif args.input == 'melspectrogram':
            print("melspectrogram")
            dataloader_generator = generate_evaluate_image_3CH_dataloaders(data_type=args.dataset, data_image=args.input)
            evaluate(args, dataloader_generator, pretrained_id)
        else:
            print("wrong")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Voice Event Detection")

    parser.add_argument('--dataset', type=str, default="Korean")
    """
    (1) English, (2) French, (3) German, (4) Spanish, (5) Korean, (6) Total
    """

    parser.add_argument('--model_dir', type=str, default="./Output/")
    parser.add_argument(f'--model', type=str, default="efficientnet")
    """
    (1) wavenet, (2) resnet50, (3) efficientnet
    """
    parser.add_argument(f'--input', type=str, default="melspectrogram")
    """
    (1) waveform, (2) f0, (3) spectrogram, (4) melspectrogram
    """
    parser.add_argument('--save_dir', type=str, default="./Output_test/")

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
