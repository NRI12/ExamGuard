"""TSM: ResNet-50 whose residual branches shift 1/8 of the channels one frame forward and 1/8 one frame back
(blockres, shift_div 8), with average consensus over segments. Kinetics-400 init from the OpenMMLab TSM-R50
checkpoint, remapped onto torchvision's ResNet-50 (no mmaction2 / mmcv needed)."""
import os
import re

import torch
import torch.nn as nn
import torchvision

K400_URL = ("https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/"
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth")


class TemporalShift(nn.Module):
    """Wraps a conv; shifts channel folds along time before it (same as mmaction2 / original TSM)."""

    def __init__(self, net, n_segment, shift_div):
        super().__init__()
        self.net, self.n_segment, self.shift_div = net, n_segment, shift_div

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w)
        fold = c // self.shift_div
        out = torch.empty_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # from the next frame
        out[:, -1, :fold] = 0
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # from the previous frame
        out[:, 0, fold:2 * fold] = 0
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return self.net(out.view(nt, c, h, w))


class TSM(nn.Module):
    """Input [B, 3, T, H, W] (same layout as X3D) -> logits [B, C]."""

    def __init__(self, num_classes, n_segment=8, shift_div=8, dropout=0.5):
        super().__init__()
        self.n_segment = n_segment
        r = torchvision.models.resnet50()
        for layer in (r.layer1, r.layer2, r.layer3, r.layer4):
            for blk in layer:
                blk.conv1 = TemporalShift(blk.conv1, n_segment, shift_div)
        r.fc = nn.Identity()
        self.backbone, self.dropout, self.fc = r, nn.Dropout(dropout), nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        b, c, t, h, w = x.shape
        assert t == self.n_segment, f"TSM built for {self.n_segment} segments, got {t}"
        x = x.transpose(1, 2).reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        return self.fc(self.dropout(self.backbone(x))).view(b, t, -1).mean(1)


def mmaction_to_torchvision(sd):
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        k = k[len("backbone."):]
        k = re.sub(r"^conv1\.conv\.", "conv1.", k)
        k = re.sub(r"^conv1\.bn\.", "bn1.", k)
        k = re.sub(r"\.downsample\.conv\.", ".downsample.0.", k)
        k = re.sub(r"\.downsample\.bn\.", ".downsample.1.", k)
        k = re.sub(r"\.conv(\d)\.bn\.", r".bn\1.", k)
        k = re.sub(r"\.conv1\.conv\.net\.", ".conv1.net.", k)
        k = re.sub(r"\.conv(\d)\.conv\.", r".conv\1.", k)
        out["backbone." + k] = v
    return out


def build_tsm(num_classes, segments=8, shift_div=8, dropout=0.5, pretrained=None):
    """pretrained: path of the OpenMMLab K400 checkpoint (downloaded there if missing), or None."""
    m = TSM(num_classes, segments, shift_div, dropout)
    if pretrained:
        if not os.path.exists(pretrained):
            os.makedirs(os.path.dirname(pretrained), exist_ok=True)
            torch.hub.download_url_to_file(K400_URL, pretrained)
        sd = mmaction_to_torchvision(torch.load(pretrained, map_location="cpu", weights_only=False)["state_dict"])
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert set(missing) == {"fc.weight", "fc.bias"} and not unexpected, (missing[:6], unexpected[:6])
    return m
