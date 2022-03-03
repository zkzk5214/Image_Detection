# -*-coding: utf-8 -*-
import torch
from torch import nn
from multibox import multibox_prior


def cls_predictor(num_inputs, num_anchors, num_classes):
    """类别预测层"""
    # 设目标类别的数量为 num_classes,则描框有 num_classes+1 个类别, 其中0类是背景
    # 以其中的每个单元为中心生成 num_anchors个 描框
    # 输出结果与输入图像的 channel 数无关
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    """边界框预测层"""
    # 每个锚框预测 4 个偏移量
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred):
    """将通道维移到最后一维,然后平铺成二维矩阵"""
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    """按列拼接"""
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    """高和宽减半块,基于VGG模块"""
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """基本网络块"""
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """选择模块组"""
    # 第⼀个是基本⽹络块
    # 第⼆个到第四个是⾼和宽减半块
    # 最后⼀个模块使⽤全局最⼤池将⾼度和宽度都降到1
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i], getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


if __name__ == "__main__":
    def forward(x, block):
        return block(x)


    # Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    # Y2 = forward(torch.zeros((2, 16, 10, 10)), bbox_predictor(16, 3))
    # print(Y1.shape, Y2.shape)

    # Y_concat = concat_preds([Y1, Y2])
    # print(Y_concat.shape)

    # Y_down_sample_blk_test = forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10))
    # print(Y_down_sample_blk_test.shape)

    # Y_base_net_test = forward(torch.zeros((2, 3, 256, 256)), base_net())
    # print(base_net())
    # print(Y_base_net_test.shape)

    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)
