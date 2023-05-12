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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_csv(file_name):
    label_name_df = pd.read_csv(file_name)
    # label_name_dict = dict(zip(df['Index'], df['Category']))
    return label_name_df

def parse_option():
    parser = argparse.ArgumentParser('Progressive Region Enhancement Network(PRENet) for training and testing')

    parser.add_argument('--batchsize', default=2, type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, default='food2k', help='food2k, food101, food500')
    parser.add_argument('--model_name', type=str, default='resnet50', help='resnet101, resnet152, senet154,densenet161')
    parser.add_argument('--image_path', type=str, default="./food/", help='path to dataset')
    parser.add_argument("--train_path", type=str, default="./meta_data/train_full.txt", help='path to training list')
    parser.add_argument("--test_path", type=str, default="./meta_data/test_full_f.txt",
                        help='path to testing list')
    parser.add_argument('--weight_path', default="./Pretrained_model/food2k_resnet50_0.0001.pth", help='path to the pretrained model')
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--checkpoint', type=str, default="./Pretrained_model/model.pth",
                        help="the path to checkpoint")
    parser.add_argument('--output_dir', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--epoch", default=200, type=int,
                        help="The number of epochs.")
    parser.add_argument("--test", action='store_true', default=True,
                        help="Testing model.")
    parser.add_argument("--label_name_file", type=str, default="./Supplementary-tables.csv", help='file which indicates the label-name pair')
    parser.add_argument("--show_result", type=bool, default=False, help='whether show the classification result')
    
    args, unparsed = parser.parse_known_args()
    return args

def test(net, criterion, batch_size, testloader, label_name_df):

    net.eval()
    test_loss = 0
    correct = 0
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

            output_concat = F.softmax(output_concat, dim=1)
            prob, top3_pos = torch.topk(output_concat.data, 5)

            total += targets.size(0)
            for i in range(targets.size(0)):
                index_batch = top3_pos[i].cpu().numpy()
                prob = prob[i].cpu().numpy().tolist()
                result_label.append(label_name_df[label_name_df['Index'].isin(list(index_batch))]['Category'].values.tolist())
                result_prob.append(prob)
                result_img_name.append(img_name[i])
            batch_corrects1 = torch.sum((top3_pos[:, 0] == targets)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == targets)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == targets)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)
            batch_idx += 1

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

    with open('./test_images_result.txt', 'w') as f:
        f.writelines(final_result)
    return test_acc, test5_acc, test_loss

def main():
    args = parse_option()
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

    net = torchvision.models.resnet50()
    net.fc = nn.Linear(2048, NUM_CATEGORIES)
    net.load_state_dict(torch.load(args.weight_path))

    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    if args.use_checkpoint:
        model = torch.load(args.checkpoint).module.state_dict()
        net.module.load_state_dict(torch.load(args.checkpoint).module.state_dict())
        print('load the checkpoint')

    if args.test:
        val_acc, val5_acc, val_loss = test(net, nn.CrossEntropyLoss(), args.batchsize, test_loader, label_name_df)
        if args.show_result:
            print('Accuracy of the network on the val images: top1 = %.5f, top5 = %.5f, test_loss = %.6f\n' % (
                    val_acc, val5_acc, val_loss))
        return


if __name__ == "__main__":
    main()
