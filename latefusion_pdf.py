#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bert import BertEncoder,BertClf
from src.models.image import ImageEncoder,ImageClf


class MultimodalLateFusionClf_pdf(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf_pdf, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)
        self.ConfidNet_txt = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.Linear(768*2, 768),
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.ConfidNet_img = nn.Sequential(
            nn.Linear(6144, 6144*2),
            nn.Linear(6144*2, 6144),
            nn.Linear(6144, 1),
            nn.Sigmoid()
        )

    def forward(self, txt, mask, segment, img, choice):
        txt_out, txt_f = self.txtclf(txt, mask, segment)
        img_out, img_f = self.imgclf(img)

        if not self.args.df:  # static fusion
            txt_img_out = 0.5 * txt_out + 0.5 * img_out
            return txt_img_out, txt_out, img_out, None, None, None, None
        else:
            txt_tcp = self.ConfidNet_txt(txt_f.clone().detach()) if choice == 'pdf_train' else self.ConfidNet_txt(txt_f)
            img_tcp = self.ConfidNet_img(img_f.clone().detach()) if choice == 'pdf_train' else self.ConfidNet_img(img_f)

            # 确保 txt_tcp 和 img_tcp 的形状一致
            if txt_tcp.shape != img_tcp.shape:
                raise ValueError(f"Shape mismatch: txt_tcp {txt_tcp.shape}, img_tcp {img_tcp.shape}")

            if choice == 'pdf_train':
                # 计算 train 动态权重
                txt_holo = torch.log(img_tcp) / (torch.log(txt_tcp * img_tcp) + 1e-8)
                img_holo = torch.log(txt_tcp) / (torch.log(txt_tcp * img_tcp) + 1e-8)
                cb_txt = txt_tcp.detach() + txt_holo.detach()
                cb_img = img_tcp.detach() + img_holo.detach()
                w_all = torch.stack((cb_txt, cb_img), 1)
            else:  # 'pdf_test'
                # 计算 test 动态权重（带RC校正）
                txt_pred = torch.softmax(txt_out, dim=1)
                img_pred = torch.softmax(img_out, dim=1)
                txt_du = torch.mean(torch.abs(txt_pred - 1 / txt_pred.shape[1]), dim=1, keepdim=True)
                img_du = torch.mean(torch.abs(img_pred - 1 / img_pred.shape[1]), dim=1, keepdim=True)
                condition = txt_du > img_du
                rc_t = torch.where(condition, torch.ones_like(txt_du), txt_du / img_du)
                rc_i = torch.where(condition, img_du / txt_du, torch.ones_like(img_du))
                cb_txt = txt_tcp * rc_t
                cb_img = img_tcp * rc_i
                w_all = torch.stack((cb_txt, cb_img), 1)

            w_all = torch.softmax(w_all, dim=1)
            w_txt, w_img = w_all[:, 0], w_all[:, 1]
            txt_img_out = w_txt.detach() * txt_out + w_img.detach() * img_out

            # 统一返回 7 个值，缺省填 None
            return txt_img_out, txt_out, img_out, txt_tcp, img_tcp, w_txt, w_img

