"""This initiates the training process"""
import argparse
import os
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from data.vimeo_dataset import VimeoDataset

from torchvision.models import vgg16
from torchvision.models import densenet
from models.lossfunc import PerceptualLoss

def setup_data(file_path_train, file_path_test, batch, num_w):
    train_set_task_2 = VimeoDataset(file_path_train)
    test_set_task_2 = VimeoDataset(file_path_test)

    trainloader_task_2 = torch.utils.data.DataLoader(train_set_task_2, batch_size=batch,
                                                     shuffle=True, num_workers=num_w)
    testloader_task_2 = torch.utils.data.DataLoader(test_set_task_2, batch_size=batch,
                                                    shuffle=False, num_workers=num_w)

    return trainloader_task_2, testloader_task_2


def setup_optimizer(backbone, task_1_net, task_2_net, config):
    backbone_params = [p for p in backbone.named_parameters()]
    task_1_params = [p for p in task_1_net.named_parameters()]
    task_2_params = [p for p in task_2_net.named_parameters()]

    optimizer_1 = optim.SGD(
        [{'params': [p[1] for p in backbone_params], 'lr': .0001},
         {'params': [p[1] for p in task_1_params], 'lr': .001}
         ],
        lr=.001, momentum=0.9, weight_decay=5e-4)

    optimizer_2 = optim.SGD(
        [{'params': [p[1] for p in backbone_params], 'lr': .0001},
         {'params': [p[1] for p in task_2_params], 'lr': .001}
         ],
        lr=.001, momentum=0.9, weight_decay=5e-4)

    return optimizer_1, optimizer_2


def train(task_2_net, trainloader_2, opt, cuda_enabled=True):
    optimizer_2 = opt
    task_2_net.train()
    #added loss function
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    relu2_2.eval()
    if(cuda_enabled):
        relu2_2.cuda()
    criterion_task_2 = PerceptualLoss(loss_network=relu2_2)
    train_loss_2 = 0.0, 0.0
    count2 = 0, 0
    psnr_running_sum = 0.0

    with tqdm(total=len(trainloader_2)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(trainloader_2):
            if (cuda_enabled):
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = task_2_net(inputs)
            loss = criterion_task_2(outputs, labels)
            #train_loss_2 += loss.item()
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            psnr_running_sum += core.get_PSNR(outputs, labels)
            count2 += 1
            pbar.update(1)
            pbar.set_description("Current training loss: %.4f, PSNR: %.4f" % ((train_loss_2 * 1. / count2),
                                 (psnr_running_sum / count2)))


def test(task_2_net, testloader_task_2):
    task_2_net.eval()
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    relu2_2.eval()
    if(cuda_enabled):
        relu2_2.cuda()
    criterion_task_2 = PerceptualLoss(loss_network=relu2_2)
    test_loss_2 = 0.0
    count, count2 = 0, 0
    with torch.no_grad():
        with tqdm(total=len(testloader_task_2)) as pbar:
            for batch_idx, (inputs, labels) in enumerate(testloader_task_2):
                if(cuda_enabled):
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = task_2_net(features)
                loss = criterion_task_2(outputs, labels)
                #test_loss_2 += loss.item()
                #psnr_running_sum += core.get_PSNR(outputs, labels)
                count2 += 1
                pbar.update(1)
                #pbar.set_description("Current testing loss: %.4f, PSNR: %.4f" % ((test_loss_2 * 1. / count2),
                #                                                                  (psnr_running_sum / count2)))
                pbar.set_description("Current testing loss: %.4f" % ((test_loss_2 * 1. / count2)))


def main():
    """Calls related methods"""
    trainloader_task_1, testloader_task_1, trainloader_task_2, testloader_task_2 = setup_data(config)

    # TODO: update this with config file
    backbone = models.DenseNetBackbone().cuda()
    backbone = torch.nn.DataParallel(backbone, device_ids=range(torch.cuda.device_count()))
    net_seg = torch.nn.DataParallel(net_seg, device_ids=range(torch.cuda.device_count()))
    net_sr = torch.nn.DataParallel(net_sr, device_ids=range(torch.cuda.device_count()))

    for epoch in range(25):
        print("Current Epoch: %d" % epoch)
        train(backbone, net_seg, net_sr, trainloader_task_1, trainloader_task_2, config)
        test(backbone, net_seg, net_sr, testloader_task_1, testloader_task_2)


if __name__ == '__main__':
    main()
