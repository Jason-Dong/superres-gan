"""This initiates the training process"""
import argparse
import os
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

'''from lib import datasets
from lib import models
from lib import core
'''

from torchvision.models import vgg16
from models.lossfunc import PerceptualLoss


def parse_args():
    """Returns args for training session"""
    parser = argparse.ArgumentParser(description='Training network')

    parser.add_argument('--cfg', default="config/segmentation+sr.yaml",
                        help='experiment config file name', required=False, type=str)

    args = parser.parse_args()
    with open(args.cfg) as f:
        config = yaml.full_load(f)

    return args, config


def setup_data(config):
    train_set_task_1 = eval("datasets." + config["DATASET"]["TASK_1"]["DATASET"] +
                            "(root=\"" + config["DATASET"]["TASK_1"]["ROOT"] + "\", " +
                            "list_path=\"" + config["DATASET"]["TASK_1"]["TRAIN_PATH"] + "\")")
    test_set_task_1 = eval("datasets." + config["DATASET"]["TASK_1"]["DATASET"] +
                           "(root=\"" + config["DATASET"]["TASK_1"]["ROOT"] + "\", " +
                           "list_path=\"" + config["DATASET"]["TASK_1"]["TEST_PATH"] + "\")")

    train_set_task_2 = eval("datasets." + config["DATASET"]["TASK_2"]["DATASET"] +
                            "(file_path=\"" + config["DATASET"]["TASK_2"]["TRAIN_PATH"] + "\")")
    test_set_task_2 = eval("datasets." + config["DATASET"]["TASK_2"]["DATASET"] +
                           "(file_path=\"" + config["DATASET"]["TASK_2"]["TEST_PATH"] + "\")")

    trainloader_task_1 = torch.utils.data.DataLoader(train_set_task_1, batch_size=config["MODEL"]["TRAIN_BATCH_SIZE"],
                                                     shuffle=True, num_workers=1)
    testloader_task_1 = torch.utils.data.DataLoader(test_set_task_1, batch_size=config["MODEL"]["TEST_BATCH_SIZE"],
                                                    shuffle=False, num_workers=1)

    trainloader_task_2 = torch.utils.data.DataLoader(train_set_task_2, batch_size=config["MODEL"]["TRAIN_BATCH_SIZE"],
                                                     shuffle=True, num_workers=1)
    testloader_task_2 = torch.utils.data.DataLoader(test_set_task_2, batch_size=config["MODEL"]["TEST_BATCH_SIZE"],
                                                    shuffle=False, num_workers=1)

    return trainloader_task_1, testloader_task_1, trainloader_task_2, testloader_task_2


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


def train(backbone, task_1_net, task_2_net, trainloader_1, trainloader_2, config):
    optimizer_1, optimizer_2 = setup_optimizer(backbone, task_1_net, task_2_net, config)

    backbone.train()
    task_1_net.train()
    task_2_net.train()

    #added loss function
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    relu2_2.eval()
    relu2_2.cuda()

    criterion_task_1 = core.CrossEntropy()
    '''
    criterion_task_2 = torch.nn.MSELoss()
    '''
    criterion_task_2 = PerceptualLoss(loss_network=relu2_2)

    train_loss, train_loss_2 = 0.0, 0.0
    count, count2 = 0, 0

    true_positive, false_positive, false_negative = 0.0, 0.0, 0.0
    psnr_running_sum = 0.0

    with tqdm(total=len(trainloader_1)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(trainloader_1):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer_1.zero_grad()
            features = backbone(inputs)
            outputs = task_1_net(features)

            loss = criterion_task_1(outputs, labels)
            loss.backward()
            optimizer_1.step()

            train_loss += loss.item()

            true_positive_batch, false_positive_batch, false_negative_batch = core.get_mIOU(outputs, labels)
            true_positive += true_positive_batch
            false_positive += false_positive_batch
            false_negative += false_negative_batch

            count += 1
            pbar.update(1)

            pbar.set_description("Current training loss: %.4f, mIOU: %.4f" % ((train_loss * 1. / count),
                                 (true_positive / (true_positive + false_positive + false_negative))))

    with tqdm(total=len(trainloader_2)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(trainloader_2):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer_2.zero_grad()
            features = backbone(inputs)
            outputs = task_2_net(features)

            loss = criterion_task_2(outputs, labels)
            loss.backward()
            optimizer_2.step()

            train_loss_2 += loss.item()

            psnr_running_sum += core.get_PSNR(outputs, labels)

            count2 += 1
            pbar.update(1)

            pbar.set_description("Current training loss: %.4f, PSNR: %.4f" % ((train_loss_2 * 1. / count2),
                                 (psnr_running_sum / count2)))


def test(backbone, task_1_net, task_2_net, testloader_task_1, testloader_task_2):
    backbone.eval()
    task_1_net.eval()
    task_2_net.eval()

    criterion_task_1 = core.CrossEntropy()
    criterion_task_2 = torch.nn.MSELoss()

    test_loss, test_loss_2 = 0.0, 0.0

    true_positive, false_positive, false_negative = 0.0, 0.0, 0.0
    psnr_running_sum, count, count2 = 0.0, 0, 0

    with torch.no_grad():
        with tqdm(total=len(testloader_task_1)) as pbar:
            for batch_idx, (inputs, labels) in enumerate(testloader_task_1):
                inputs, labels = inputs.cuda(), labels.cuda()

                features = backbone(inputs)
                outputs = task_1_net(features)

                loss = criterion_task_1(outputs, labels)
                test_loss += loss.item()

                true_positive_batch, false_positive_batch, false_negative_batch = core.get_mIOU(outputs, labels)
                true_positive += true_positive_batch
                false_positive += false_positive_batch
                false_negative += false_negative_batch

                count += 1

                pbar.update(1)
                pbar.set_description("Current testing loss: %.4f, mIOU: %.4f" % ((test_loss * 1. / count),
                                     (true_positive / (true_positive + false_positive + false_negative))))

        with tqdm(total=len(testloader_task_2)) as pbar:
            for batch_idx, (inputs, labels) in enumerate(testloader_task_2):
                inputs, labels = inputs.cuda(), labels.cuda()

                features = backbone(inputs)
                outputs = task_2_net(features)

                loss = criterion_task_2(outputs, labels)
                test_loss_2 += loss.item()

                psnr_running_sum += core.get_PSNR(outputs, labels)
                count2 += 1

                pbar.update(1)
                pbar.set_description("Current testing loss: %.4f, PSNR: %.4f" % ((test_loss_2 * 1. / count2),
                                                                                  (psnr_running_sum / count2)))


def main():
    """Calls related methods"""
    args, config = parse_args()
    trainloader_task_1, testloader_task_1, trainloader_task_2, testloader_task_2 = setup_data(config)

    # TODO: update this with config file
    backbone = models.DenseNetBackbone().cuda()
    net_seg = models.SegmentationModelDenseNet().cuda()
    net_sr = models.BasicSuperResolution().cuda()

    backbone = torch.nn.DataParallel(backbone, device_ids=range(torch.cuda.device_count()))
    net_seg = torch.nn.DataParallel(net_seg, device_ids=range(torch.cuda.device_count()))
    net_sr = torch.nn.DataParallel(net_sr, device_ids=range(torch.cuda.device_count()))

    for epoch in range(25):
        print("Current Epoch: %d" % epoch)
        train(backbone, net_seg, net_sr, trainloader_task_1, trainloader_task_2, config)
        test(backbone, net_seg, net_sr, testloader_task_1, testloader_task_2)


if __name__ == '__main__':
    main()
