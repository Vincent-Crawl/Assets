# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn#神经网络模块（nn）

import timm.models.vision_transformer#vison_transformer


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()#调用父类nn.Module构造方法

    def forward(self, x):
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):#继承Visiontransformer
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
            # norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)#使输入的数据更均匀，避免过大或过小的值影响模型的学习
            # self.fc_norm = Identity()

            del self.norm  # remove the original norm

    def forward_head(self, x, pre_logits: bool = False):
        # if self.global_pool:
        #     x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        # x = self.head_drop(x)
        return self.head(x)#self.head 层通常是网络的最后一层，负责输出网络的预测结果
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)#将原始的图片输入转化为适合 Transformer 模型处理的序列形式

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)#通过模型的所有层进行前向传播

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)#层归一化
        else:
            x = self.norm(x)#层归一化
            outcome = x[:, 0]
        return outcome


# def vit_nano_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


#😀这些函数的目的都是创建和返回一个特定规模的 VisionTransformer 模型
#控制模型的大小、复杂度和行为

#patch_size:分解成16*16像素的块,embed_dim,depth:嵌入维度,深度,理论上越大模型表现越好
#num_heads:self-attention操作时并行的子任务数量 
#mlp_ratio:feed-forward network隐藏层的大小
#qkv_bias: 决定是否在计算 Query、Key 和 Value 的时候添加偏置项
#norm_layer:使用何种归一化层

def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model