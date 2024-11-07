
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a_q: nn.Module, w_b_q: nn.Module, 
                 w_a_v: nn.Module, w_b_v: nn.Module, r: int, alpha: int, dim: int):
        super().__init__()
        self.w = w
        self.w_a_q = w_a_q
        self.w_b_q = w_b_q
        self.w_a_v = w_a_v
        self.w_b_v = w_b_v
        self.r = r
        self.alpha = alpha
        self.dim = dim

    def forward(self, x):
        val = self.w(x)
        val[:, :, :self.dim] += (self.alpha / self.r) * self.w_b_q(self.w_a_q(x))
        val[:, :, self.dim*2:] += (self.alpha / self.r) * self.w_b_v(self.w_a_v(x))
        return val

class LoRA_ViT(nn.Module):
    def __init__(self, vit_model, r: int, alpha: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()
        base_vit_dim = vit_model.blocks[0].attn.qkv.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        for param in vit_model.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(vit_model.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv

            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)

            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = _LoRALayer(w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v, r, alpha, dim)

        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_intermediate_layers(self, x: torch.Tensor, n, reshape=False, return_class_token=False, norm=True):
        return self.lora_vit.get_intermediate_layers(x, n, reshape, return_class_token, norm)

class LinearHead(nn.Module):
    def __init__(self, dino, r, in_features=1920, out_features=101, bias=True):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if r == 0:
            for param in dino.parameters():
                param.requires_grad = False
            self.lora_dino = dino
        elif r == 255:
            self.lora_dino = dino
        else:
            self.lora_dino = LoRA_ViT(dino, r=r, alpha=15, num_classes=0)

    def forward(self, imgs, num_layers):
        feature = self.lora_dino.get_intermediate_layers(imgs, num_layers, return_class_token=True)
        outputs = []
        for patch_idx in range(feature[0][0].shape[1]):
            patch_features = torch.cat([feature[layer_idx][0][:, patch_idx, :] for layer_idx in range(num_layers)], dim=-1)
            patch_output = self.linear(patch_features)
            outputs.append(patch_output)

        outputs = torch.stack(outputs, dim=1)
        return outputs
