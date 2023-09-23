from __future__ import print_function
from PIL import Image
import torch.utils.data as data
import os
import PIL
import argparse
from tqdm import tqdm
import torch.optim as optim
from data_loader_resnet50 import load_data
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import re
from utils import load_model
import torchvision
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pandas as pd
from inceptionresnetv2 import inceptionresnetv2
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_csv(file_name):
    # load label ID and label name
    label_name_df = pd.read_csv(file_name)
    # label_name_dict = dict(zip(df['Index'], df['Category']))
    return label_name_df

def parse_option():
    parser = argparse.ArgumentParser('Progressive Region Enhancement Network(PRENet) for testing')

    parser.add_argument('--batchsize', default=2, type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='food2k', help='food2k, food101, food500')
    parser.add_argument('--model_name', type=str, default='resnet50', help='resnet50, densenet161, inceptionresnetv2')
    parser.add_argument('--image_path', type=str, default="./food/", help='path to dataset')
    parser.add_argument("--train_path", type=str, default="./meta_data/train_full.txt", help='path to training list')
    parser.add_argument("--test_path", type=str, default="./meta_data/test_full_f.txt", help='path to testing list')
    parser.add_argument('--weight_path', default="./Pretrained_model/food2k_resnet50_0.0001.pth", help='path to the pretrained model')
    parser.add_argument("--label_name_file", type=str, default="./Supplementary-tables.csv", help='file which indicates the label-name pair')
    parser.add_argument("--show_result", action='store_true', default=True, help='whether show the classification result')
    parser.add_argument("--test_per_label", action='store_true', default=False,
                        help="Testing model.")    
    args, unparsed = parser.parse_known_args()
    return args

def test(args, net, criterion, batch_size, testloader, label_name_df):
    '''
    	calculate the accuracy of all test data
    '''
    net.eval()
    test_loss = 0
    total = 0
    idx = 0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0

    batch_idx = 0
    result_label = []
    result_prob = []
    result_img_name = []
    for (inputs, targets, img_name) in tqdm(testloader):
        idx = batch_idx
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            output_concat = net(inputs)

            loss = criterion(output_concat, targets)
            test_loss += loss.item()
            
            # get the probability of prediction result
            output_concat = F.softmax(output_concat, dim=1)
            # get the top-5 probability and label ID of prediction result
            prob, top3_pos = torch.topk(output_concat.data, 5)

            total += targets.size(0)
            for i in range(targets.size(0)):
                index_batch = top3_pos[i].cpu().numpy()
                prob = prob[i].cpu().numpy().tolist()
                # get the label name based on the label ID
                result_label.append(label_name_df[label_name_df['Index'].isin(list(index_batch))]['Category'].values.tolist())
                
                result_prob.append(prob)
                result_img_name.append(img_name[i])
            
            #  if the prediction result is the same as the target, count the number
            batch_corrects1 = torch.sum((top3_pos[:, 0] == targets)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == targets)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == targets)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)
            batch_idx += 1
    
    # record the result which look like 
    # "0/160.jpg Grilled beef tongue 0.1397 Crepe 0.1287 Salmon ﬂank sinew 0.1127 Bacon bibimbap 0.0361 Tuna sashimi 0.0233"
    final_result = []
    for i in range(len(result_img_name)):
        temp_list = [result_img_name[i]]

        for l,p in zip(result_label[i], result_prob[i]):
            temp_list += [l, '{:.4f}'.format(p)]
        # print(temp_list)
        final_result.append(' '.join(temp_list)+'\n')
    
    #print(final_result)
    test_acc = val_corrects1 / total
    test5_acc = val_corrects5 / total

    test_loss = test_loss / (idx + 1)

    with open('./test_images_result_{}.txt'.format(args.model_name), 'w') as f:
        f.writelines(final_result)
    return test_acc, test5_acc, test_loss

def test_per_label(args, net, criterion, batch_size, testloader, label_name_df):
    '''
        calculate the accuracy of test data that have the same label
    '''
    net.eval()
    idx = 0
    test_loss = 0
    batch_idx = 0
    
    # record data based on the label ID
    total = defaultdict(int)
    val_corrects1 = defaultdict(int)
    val_corrects2 = defaultdict(int)
    val_corrects5 = defaultdict(int)

    
    result_label = []
    result_prob = []
    result_img_name = []
    
    for (inputs, targets, img_name) in tqdm(testloader):
        idx = batch_idx
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            output_concat = net(inputs)

            loss = criterion(output_concat, targets)
            test_loss += loss.item()

            output_concat = F.softmax(output_concat, dim=1)
            prob, top3_pos = torch.topk(output_concat.data, 5)

            
            for i in range(targets.size(0)):
                index_batch = top3_pos[i].cpu().numpy()
                target = targets[i].item()
                total[target] += 1
                prob = prob[i].cpu().numpy().tolist()
                # get the label name based on the label ID
                result_label.append(label_name_df[label_name_df['Index'].isin(list(index_batch))]['Category'].values.tolist())
                # get the probability of prediction result
                result_prob.append(prob)
                result_img_name.append(img_name[i])
                
                # if the prediction result is the same as the target, plus 1
                if index_batch[0] == target:
                    val_corrects1[target] += 1
                    val_corrects2[target] += 1
                    val_corrects5[target] += 1
                if index_batch[1] == target:
                    val_corrects2[target] += 1
                    val_corrects5[target] += 1
                if index_batch[2] == target:
                    val_corrects5[target] += 1
                if index_batch[3] == target:
                    val_corrects5[target] += 1
                if index_batch[4] == target:
                    val_corrects5[target] += 1
                
            batch_idx += 1

    # record the result which look like 
    # "0/160.jpg Grilled beef tongue 0.1397 Crepe 0.1287 Salmon ﬂank sinew 0.1127 Bacon bibimbap 0.0361 Tuna sashimi 0.0233"
    final_result = []
    for i in range(len(result_img_name)):
        temp_list = [result_img_name[i]]

        for l,p in zip(result_label[i], result_prob[i]):
            temp_list += [l, '{:.4f}'.format(p)]
        final_result.append(' '.join(temp_list)+'\n')
        
    test_loss = test_loss / (idx + 1)
    
    # record the accuracy result per label which look like
    # resnet50 Grilled beef tongue  0.5 1
    # resnet50 Flavored snail meat  0.8 1
    # ...
    final_acc = []
    for target_ in val_corrects5:
        # print(target_, val_corrects1[target_], val_corrects5[target_], total[target_])
        test_acc = val_corrects1[target_] / total[target_]
        test5_acc = val_corrects5[target_] / total[target_]
        final_acc.append([args.model_name, label_name_df[label_name_df['Index'] == target_]['Category'].values[0], test_acc, test5_acc])
        print('Accuracy of the network on the val images: label is %s, top1 = %.5f, top5 = %.5f, test_loss = %.6f\n' % (str(target_), test_acc, test5_acc, test_loss))


    df_acc = pd.DataFrame(data=final_acc, columns=['Model', 'label', 'Top-1 test Acc', 'Top-5 test Acc'])
    df_acc.to_csv('./test_acc_result_per_label_{}.csv'.format(args.model_name))
    with open('./test_images_result_per_label_{}.txt'.format(args.model_name), 'w') as f:
        f.writelines(final_result)

    
def main():
    args = parse_option()
    print(args)
    train_dataset, train_loader, test_dataset, test_loader = \
        load_data(image_path=args.image_path, train_dir=args.train_path, test_dir=args.test_path,batch_size=args.batchsize)
    print('Data Preparation : Finished')

    label_name_df = read_csv(args.label_name_file)
    if args.dataset == "food101":
        NUM_CATEGORIES = 101
    elif args.dataset == "food500":
        NUM_CATEGORIES = 500
    elif args.dataset == "food2k":
        NUM_CATEGORIES = 2000
    
    if args.model_name == 'resnet50':
        net = torchvision.models.resnet50()
        net.fc = nn.Linear(2048, NUM_CATEGORIES)
        net.load_state_dict(torch.load(args.weight_path))
    elif args.model_name == 'densenet161':
        net = torchvision.models.densenet161()
        net.classifier = nn.Linear(2208, NUM_CATEGORIES)
        state_dict_ = torch.load(args.weight_path)
        for key in list(state_dict_.keys()):
        	state_dict_[key.replace('module.', '')] = state_dict_.pop(key)
        net.load_state_dict(state_dict_)   
    elif args.model_name == 'inceptionresnetv2':
        net = inceptionresnetv2(num_classes=NUM_CATEGORIES, pretrained=None)
        state_dict_ = torch.load(args.weight_path)
        for key in list(state_dict_.keys()):
        	state_dict_[key.replace('module.', '')] = state_dict_.pop(key)
        #print(list(state_dict_.keys()))
        net.load_state_dict(state_dict_)

    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    if not args.test_per_label:
        val_acc, val5_acc, val_loss = test(args, net, nn.CrossEntropyLoss(), args.batchsize, test_loader, label_name_df)
        if args.show_result:
            print('Accuracy of the network on the val images: top1 = %.5f, top5 = %.5f, test_loss = %.6f\n' % (val_acc, val5_acc, val_loss))
    else:
        test_per_label(args, net, nn.CrossEntropyLoss(), args.batchsize, test_loader, label_name_df)


if __name__ == "__main__":
    main()
