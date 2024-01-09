import random

import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from .attention import Transformer, WindowAttention
from .layer import ln2d
from functools import partial


def cosine_similarity(a, b, eps=1e-5):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-5):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2, tem=4):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha=1.0, beta=6.0, warmup=20, temperature=4):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

    def forward(self, logits_student, logits_teacher, target, epoch):
        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        return loss_dkd


class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return self.beta * (self.temperature ** 2) * soft_loss
        hard_loss = self.cross_entropy_loss(student_output, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


"""Knowledge Distillation via Flow Matching"""


def hyp_split(loss_type: str):
    hyp = loss_type.split("_")[1:]
    hyp_dict = {}
    for i, j in zip(hyp[::2], hyp[1::2]):
        hyp_dict[i] = float(j)
    return hyp_dict


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# Copyright (c) OpenMMLab. All rights reserved.


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, channel, (1, 1), (1, 1), (0, 0), bias=False),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=False, groups=channel))

    def forward(self, x):
        y = self.block(x)
        return y


def Meta_Encoder(type="conv", window_size=4, number=2):
    if type == "conv":
        module = lambda student_channel: nn.Sequential(nn.SiLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1),
                                                                 bias=False),
                                                       nn.GroupNorm(1, student_channel),
                                                       nn.SiLU(inplace=True),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1),
                                                                 bias=False))
    elif type == "conv_mobilenet":
        module = lambda student_channel: nn.Sequential(nn.ReLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1),
                                                                 (0, 0), bias=False),
                                                       nn.GroupNorm(1, student_channel),
                                                       nn.ReLU(inplace=False),
                                                       nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                                                                 (1, 1), groups=student_channel, bias=False))
    elif type == "resconv":
        class ConvBlock(nn.Module):
            def __init__(self, student_channel):
                super().__init__()
                self.student_channel = student_channel
                self.blcok1 = nn.Sequential(
                    nn.SiLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                              (1, 1),
                              bias=False),
                    nn.GroupNorm(1, student_channel)
                )
                self.blcok2 = nn.Sequential(
                    nn.SiLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (3, 3), (1, 1),
                              (1, 1),
                              bias=False),
                    nn.GroupNorm(1, student_channel)
                )

            def forward(self, x):
                x = self.blcok1(x) + x
                x = self.blcok2(x) + x
                return x

        module = lambda student_channel: ConvBlock(student_channel)
    elif type == "transformer":
        attn1 = partial(WindowAttention, window_size=window_size, shifted=True)
        module = lambda student_channel: Transformer(student_channel,
                                                     heads=4, dim_head=64,
                                                     attn=attn1, f=partial(nn.Conv2d, kernel_size=1),
                                                     dropout=0, norm=ln2d)
        return module
    elif type == "mlp":
        class MLP(nn.Module):
            def __init__(self, student_channel, if_norm=False):
                super().__init__()
                self.module = nn.Sequential(
                    nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(student_channel, student_channel, (1, 1), (1, 1), (0, 0), bias=False))
                self.norm = ln2d(student_channel)
                self.if_norm = if_norm

            def forward(self, x):
                if self.if_norm:
                    return self.norm(self.module(x) + x)
                else:
                    return self.module(x) + x

        module = lambda student_channel: nn.Sequential(
            *[MLP(student_channel, if_norm=True) for _ in range(number)])
    else:
        raise NotImplementedError
    return module


def lower_power(n):
    exponent = math.floor(math.log2(n))
    return 2 ** exponent


class FlowAlignModule(nn.Module):
    def __init__(self, teacher_channel, student_channel, type="feature_based", teacher_size=None,
                 student_size=None, sampling=8, dirac_ratio=1., weight=1.0, apply_warmup=False,
                 loss_type=None, encoder_type=None, apply_model_based_timestep=False,
                 apply_pred_hat_x=False, second_kd=False, number=2):
        super().__init__()
        self.type = type
        self.second_kd = second_kd
        assert self.type in ["feature_based", "logit_based"]
        if self.type == "feature_based":
            assert teacher_size is not None and student_size is not None, \
                "For feature-based distillation, FlowAlignModule should " \
                "know the feature map size of teacher intermediate output" \
                " and student intermediate output"
        self.teacher_channel = teacher_channel
        self.student_channel = student_channel
        self.teacher_size = teacher_size
        self.student_size = student_size
        self.time_embedding = student_channel
        self.sampling = sampling
        self.apply_warmup = apply_warmup
        self.apply_model_based_timestep = apply_model_based_timestep
        self.apply_pred_hat_x = apply_pred_hat_x
        self.weight = weight
        print("sampling is:", sampling, "\tdirac ratios is:", dirac_ratio,
              "\tapply_warmup is:", apply_warmup, "\tweight is:", weight, "\n"
                                                                          "loss type is:", loss_type,
              "\tencoder type is", encoder_type, "\tapply model based timestep is:", apply_model_based_timestep,
              "\tsecond_kd is:", second_kd)
        self.dirac_ratio = 1 - dirac_ratio
        if self.type == "feature_based":
            raise NotImplementedError
        else:
            self.align_loss = DISTLoss()
            self.lowermodule = nn.Identity()
            self.studentmodule = nn.Identity()
            self.flowembedding = Meta_Encoder(encoder_type if encoder_type is not None else "conv",
                                              window_size=7 if student_size % 7 == 0 else 2,
                                              number=number)(student_channel)
            self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(student_channel, teacher_channel))
            if self.second_kd:
                self.copy_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                             nn.Flatten(),
                                             nn.Linear(student_channel, teacher_channel))

            if self.apply_model_based_timestep:
                self.time_embed = nn.ModuleList([ResBlock(student_channel) for _ in
                                                 range(self.sampling)])
            else:
                self.time_embed = nn.Sequential(
                    nn.Linear(self.student_channel, self.student_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.student_channel, self.student_channel))

        if loss_type != None:
            loss_type: str
            if loss_type == "dkd" or loss_type.startswith("dkd"):
                if teacher_size is not None and teacher_size % 7 == 0 and len(loss_type) == 3:
                    self.align_loss = DKDLoss(alpha=0.5, beta=0.5, warmup=5, temperature=1)
                    print("Apply in ImageNet", teacher_size, student_size, teacher_channel, student_channel)
                elif len(loss_type) == 3:
                    self.align_loss = DKDLoss(alpha=1.0, beta=2.0, warmup=20, temperature=4)
                else:
                    self.align_loss = DKDLoss(**hyp_split(loss_type))

            elif loss_type == "dist" or loss_type.startswith("dist"):
                if len(loss_type) == 4:
                    self.align_loss = DISTLoss(beta=2, gamma=2, tem=1)
                else:
                    self.align_loss = DISTLoss(**hyp_split(loss_type))

            elif loss_type == "mse" or loss_type.startswith("mse"):
                self.align_loss = nn.MSELoss(reduction="mean")

            elif loss_type == "kl" or loss_type.startswith("kl"):
                if len(loss_type) == 2:
                    self.align_loss = KDLoss(temperature=4, alpha=0, beta=1)
                else:
                    self.align_loss = KDLoss(**hyp_split(loss_type))
            else:
                raise NotImplementedError

    def forward(self, student_feature, teacher_feature, inference_sampling=4, **kwargs):
        if student_feature.ndim == 2:
            student_feature = student_feature[..., None, None].expand(-1, -1, 7, 7)
        if self.weight == 0:
            return torch.Tensor([0.]).to(student_feature.device), student_feature

        student_feature = self.studentmodule(student_feature)

        if teacher_feature is not None:
            _len_dirac = int(self.dirac_ratio * teacher_feature.shape[0])
            teacher_feature[:_len_dirac][torch.randperm(_len_dirac, device=student_feature.device)] \
                = teacher_feature[:_len_dirac].clone()
            teacher_feature = teacher_feature.contiguous()

        if self.training:
            """
            Random Sampling Aware
            """
            if isinstance(self.align_loss, DKDLoss):
                align_loss = lambda s, t: self.align_loss(s, t, target=kwargs["target"], epoch=kwargs["epoch"])
            else:
                align_loss = self.align_loss

            if self.type == "feature_based":
                inference_sampling = [1, 2, 4, 8, 16]
            else:
                inference_sampling = [self.sampling]
            inference_sampling = np.random.choice(inference_sampling, 1)[0]
            if self.apply_warmup:
                inference_sampling = min(inference_sampling, int((kwargs["epoch"] + 10) / 10))
                inference_sampling = lower_power(inference_sampling)
            indices = reversed(range(1, inference_sampling + 1))
            x = student_feature
            total_velocity = []
            loss = 0.
            t_output_feature = self.lowermodule(teacher_feature)
            if self.type != "feature_based":
                outputs = []
            for i in indices:
                _weight = self.weight / inference_sampling
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                if not self.apply_model_based_timestep:
                    _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                 self.time_embedding, 1,
                                                                                                 1)
                    embed_x = x + _t_embed
                else:
                    _t_embed = self.time_embed[i - 1](x)
                    embed_x = x + _t_embed
                _velocity = self.flowembedding(embed_x)
                if self.apply_pred_hat_x:
                    _velocity = student_feature - _velocity
                x = x - _velocity / inference_sampling
                total_velocity.append(_velocity)
                if _weight != 0:
                    if self.type == "feature_based":
                        loss += align_loss(self.fc(student_feature) - t_output_feature,
                                           self.fc(_velocity)).mean() * _weight
                    else:
                        output = self.fc(student_feature - _velocity)
                        outputs.append(output)
                        loss += ((align_loss(output, t_output_feature)) + kwargs['cls_loss'](output, kwargs[
                            "target"])).mean() * _weight
            if self.type != "feature_based":
                x = torch.stack(outputs, 0).mean(0)

            if self.second_kd:
                lightweight_x = self.copy_fc(student_feature)
                loss += ((align_loss(lightweight_x, x.detach())) + kwargs['cls_loss'](lightweight_x, kwargs[
                    "target"])).mean() * self.weight
            return loss, x

        else:
            if self.second_kd:
                lightweight_x = self.copy_fc(student_feature)
                return torch.Tensor([0.]).to(lightweight_x.device), lightweight_x
            else:
                x = student_feature
                indices = reversed(range(1, inference_sampling + 1))
                if self.type != "feature_based":
                    outputs = []
                for i in indices:
                    _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                    if not self.apply_model_based_timestep:
                        _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                     self.time_embedding,
                                                                                                     1,
                                                                                                     1)
                        embed_x = x + _t_embed
                    else:
                        _t_embed = self.time_embed[int(i / inference_sampling * self.sampling) - 1](x)
                        embed_x = x + _t_embed
                    _velocity = self.flowembedding(embed_x)
                    if self.apply_pred_hat_x:
                        _velocity = student_feature - _velocity
                    x = x - _velocity / inference_sampling
                    if self.type != "feature_based":
                        output = self.fc(student_feature - _velocity)
                        outputs.append(output)
                if self.type != "feature_based":
                    x = torch.stack(outputs, 0).mean(0)
                return torch.Tensor([0.]).to(x.device), x


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)
    gold = gold.to(pred.device)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1)).to(pred.device)  # 0.0111111
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1).long(), value=1. - smoothing)  # 0.9
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.nn.functional.kl_div(input=log_prob, target=one_hot, reduction='none').sum(dim=-1).mean()


class OnlineFlowAlignModule(FlowAlignModule):

    def forward(self, student_feature, inference_sampling=4, **kwargs):
        if self.weight == 0:
            return torch.Tensor([0.]).to(student_feature.device), student_feature

        student_feature = self.studentmodule(student_feature)

        if self.training:
            """
            Random Sampling Aware
            """
            if isinstance(self.align_loss, DKDLoss):
                align_loss = lambda s, t: self.align_loss(s, t, target=kwargs["target"], epoch=kwargs["epoch"])
            else:
                align_loss = self.align_loss

            if self.type == "feature_based":
                inference_sampling = [1, 2, 4, 8, 16]
            else:
                inference_sampling = [self.sampling]
            inference_sampling = np.random.choice(inference_sampling, 1)[0]
            if self.apply_warmup:
                inference_sampling = min(inference_sampling, int((kwargs["epoch"] + 10) / 10))
                inference_sampling = lower_power(inference_sampling)
            indices = reversed(range(1, inference_sampling + 1))
            x = student_feature
            total_velocity = []
            loss = 0.
            if self.type != "feature_based":
                outputs = []
            for i in indices:
                _weight = self.weight / inference_sampling
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                if not self.apply_model_based_timestep:
                    _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                 self.time_embedding, 1,
                                                                                                 1)
                    embed_x = x + _t_embed
                else:
                    _t_embed = self.time_embed[i - 1](x)
                    embed_x = x + _t_embed
                _velocity = self.flowembedding(embed_x)
                if self.apply_pred_hat_x:
                    _velocity = student_feature - _velocity
                x = x - _velocity / inference_sampling
                total_velocity.append(_velocity)
                if _weight != 0:
                    if self.type == "feature_based":
                        pass
                    else:
                        output = self.fc(student_feature - _velocity)
                        outputs.append(output)
                        loss += smooth_crossentropy(output, kwargs["target"]).mean() * _weight
            if self.type != "feature_based":
                x = torch.stack(outputs, 0).mean(0)
                s_t = x.clone().detach()
                for i in range(len(outputs)):
                    loss += align_loss(outputs[i], s_t).mean() * _weight
            return loss, x

        else:
            x = student_feature
            indices = reversed(range(1, inference_sampling + 1))
            if self.type != "feature_based":
                outputs = []
            for i in indices:
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                if not self.apply_model_based_timestep:
                    _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],
                                                                                                 self.time_embedding, 1,
                                                                                                 1)
                    embed_x = x + _t_embed
                else:
                    _t_embed = self.time_embed[int(i / inference_sampling * self.sampling) - 1](x)
                    embed_x = x + _t_embed
                _velocity = self.flowembedding(embed_x)
                if self.apply_pred_hat_x:
                    _velocity = student_feature - _velocity
                x = x - _velocity / inference_sampling
                if self.type != "feature_based":
                    output = self.fc(student_feature - _velocity)
                    outputs.append(output)
            if self.type != "feature_based":
                x = torch.stack(outputs, 0).mean(0)
            return torch.Tensor([0.]).to(x.device), x
