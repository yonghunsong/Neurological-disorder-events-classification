import argparse
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from models import WavenetClassifier
from data_utils import generate_cv_dataloaders, get_test_dataloader, generate_fold_to_cv_dataloaders
import pandas as pd
from evaluation_indicator import result_Confusion_ROC_AUC_save

__MODEL__ = {
    'wavenet' : WavenetClassifier,
}


def train(args,model,trainloader,criterion,optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for idx, (data, labels) in enumerate(trainloader):
        data = data.to(args.device, dtype=torch.float32)
        labels = labels.type(torch.long).to(args.device)
        labels = labels.squeeze(dim=1)
        
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.squeeze()).sum().item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = (correct_predictions / total_predictions) * 100.0

    return epoch_loss, epoch_accuracy
    
def evaluate(args, model, testloader, pretrained_id, cv, epoch):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred, y_probability = [], [], []

    for idx, (data, labels) in enumerate(testloader):
        data = data.to(args.device, dtype=torch.float32)
        labels = labels.type(torch.long).to(args.device)
        labels = labels.squeeze(dim=1)

        with torch.no_grad():
            outputs = model.forward(data)

        _, predicted = torch.max(outputs, 1)
        
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.squeeze()).sum().item()

        outputs_values = outputs.tolist()[0]
        y_probability.append(outputs_values)
        y_true.append(labels.squeeze().numpy().astype(int).item(0))
        y_pred.append(predicted.numpy().astype(int).item(0))

    if epoch + 1 == args.num_epochs:
        result_Confusion_ROC_AUC_save(args, y_probability, y_true, args.num_classes, pretrained_id, cv)
        
    val_accuracy = (correct_predictions / total_predictions) * 100.0
    
    return val_accuracy, y_true, y_pred
                        

def main(args):
    

    criterion = nn.CrossEntropyLoss()
    pretrained_id = time.time()
    print("load data : " + args.dataset)
    cv_dataloader_generator = generate_fold_to_cv_dataloaders(data_type=args.dataset,
                                                   num_fold=args.num_cv,
                                                   batch_size=args.batch_size)

    cv_acc = []

    for cv, dataloader in enumerate(cv_dataloader_generator):
        print("-" * 80)
        print("cv : {:02d} ( / {:02d})".format(cv + 1, args.num_cv))
        print("# of training data number:", len(dataloader['train'])*args.batch_size)
        print("# of validation data number:", len(dataloader['valid']))
        print("load model : " + args.model)
        start_time = time.time()
        model = __MODEL__[args.model](num_classes=args.num_classes,
                                      window_size=args.window_size,
                                      batch_size=args.batch_size)
        
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters()) 

        for epoch in range(args.num_epochs):

            train_loss, train_acc = train(args,model,dataloader['train'],criterion,optimizer)
            print("[{:02d}/{:02d}] loss : {:.3f}, train acc : {:.3f}".format(epoch + 1, args.num_epochs, train_loss, train_acc), end='')

            valid_acc, temp_y_true, temp_y_pred = evaluate(args,model,dataloader['valid'], pretrained_id, cv, epoch)
            print(", valid acc : {:.3f}".format(valid_acc))

            if epoch + 1 == args.num_epochs:
                cv_acc.append(valid_acc)

                if args.save_dir is not None:
                    save_path = os.path.join(args.save_dir, f'{args.model}_{int(pretrained_id)}_{cv}.pth')
                    print(f"pretrained model saved : {save_path}")
                    torch.save({'args': args,
                                'cv' : cv,
                                'cv_acc' : valid_acc,
                                'model_state_dict' : model.state_dict()}, save_path)

        save_excel_path = os.path.join(args.save_dir, f'{args.model}_{int(pretrained_id)}_{cv}.xlsx')
        validation_value = pd.DataFrame({'True label': temp_y_true, 'Prediction label': temp_y_pred})
        validation_value.to_excel(save_excel_path)
        
        print("cv : {:02d} ( / {:02d}) acc : {:.3f}, time : {:.2f} sec".format(cv + 1, args.num_cv, valid_acc, time.time() - start_time))
        print("-" * 80)

    print(cv_acc)
    print("mean cv acc : {:.5f}".format(sum(cv_acc) / args.num_cv))
    print("cv accs : {}".format(list(map(lambda x : np.round(x, 4), cv_acc))))


    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Voice Event Detection")
    parser.add_argument('--dataset', type=str, default="five_fold")
    parser.add_argument('--model', type=str, default="wavenet")
    
    parser.add_argument('--save_dir', type=str, default="./pretrained/")

    parser.add_argument('--num_cv', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=4000)

    args = parser.parse_args()

    mp.set_start_method('spawn')

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("GPU")
    else:
        args.device = torch.device('cpu')
        print("CPU")

    args.num_classes = 4

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    print(args)
    main(args)
