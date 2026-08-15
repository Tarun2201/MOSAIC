import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from resnet18_self import ResNet18
from collections import OrderedDict

class Res18(nn.Module):
    """ResNet18 with projection head for contrastive learning.
    Input: (b, 3, 224, 224)
    Output: feature (b, 512), out (b, 256)
    """
    def __init__(self):
        super(Res18, self).__init__()
        resnet=resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        #encoder
        self.f = nn.Sequential(*list(resnet.children())[:-1])  # Output: (b, 512, 1, 1)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))
        
    def forward(self, x):
        # x: (b, 3, 224, 224)
        x = self.f(x)  # (b, 512, 1, 1)
        feature = torch.flatten(x, start_dim=1)  # (b, 512)
        out = self.g(feature)  # (b, 256)
        
        return feature, out
    
    def load_pretrain_weight(self, pretrain_path):

        print("Model restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)["model_state_dict"]
        state_dict_init = self.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            new_state_dict[k_0] = v
        self.load_state_dict(new_state_dict, strict=False)

class Res18_Classifier(nn.Module):
    """ResNet18-based multi-scale classifier with CAM generation.
    
    Input: (b, 3, 224, 224)
    Outputs:
        - logits_collect: List of 4 tensors, each (b, num_classes)
        - map_collect: List of 4 tensors, each (b, num_classes, 224, 224) [if return_maps=True]
    """
    def __init__(self, num_classes=1, pretrain_path=None):
        super(Res18_Classifier, self).__init__()

        self.f = ResNet18(norm_layer=nn.InstanceNorm2d)  # returns 4 intermediate features
        self.gap = nn.AdaptiveAvgPool2d(1)  # (b, c, h, w) -> (b, c, 1, 1)
        self.in3 = nn.InstanceNorm2d(3, affine=True)
        self.ic1 = nn.Conv2d(64, num_classes, kernel_size=1)  # (b, 64, 56, 56) -> (b, num_classes, 56, 56)
        self.ic2 = nn.Conv2d(128, num_classes, kernel_size=1)  # (b, 128, 28, 28) -> (b, num_classes, 28, 28)
        self.ic3 = nn.Conv2d(256, num_classes, kernel_size=1)  # (b, 256, 14, 14) -> (b, num_classes, 14, 14)
        self.ic4 = nn.Conv2d(512, num_classes, kernel_size=1)  # (b, 512, 7, 7) -> (b, num_classes, 7, 7)

        if pretrain_path is not None:
            self.load_pretrain_weight(pretrain_path)
        else:
            print("Model from scratch")

    def forward(self, x, return_maps=True, return_l1_features=False):
        # x: (b, 3, 224, 224)
        batch_size, _, H, W = x.shape
        x = self.in3(x)  # (b, 3, 224, 224)
        
        # ResNet18 feature extraction at multiple scales
        # l1: (b, 64, 56, 56), l2: (b, 128, 28, 28), l3: (b, 256, 14, 14), l4: (b, 512, 7, 7)
        l1, l2, l3, l4 = self.f(x)

        # Convert to class activation maps via 1x1 convolutions
        l1_map = self.ic1(l1)  # (b, num_classes, 56, 56)
        l2_map = self.ic2(l2)  # (b, num_classes, 28, 28)
        l3_map = self.ic3(l3)  # (b, num_classes, 14, 14)
        l4_map = self.ic4(l4)  # (b, num_classes, 7, 7)

        # Global average pooled logits (for classification losses)
        l1_logits = torch.flatten(self.gap(l1_map), start_dim=1)  # (b, num_classes)
        l2_logits = torch.flatten(self.gap(l2_map), start_dim=1)  # (b, num_classes)
        l3_logits = torch.flatten(self.gap(l3_map), start_dim=1)  # (b, num_classes)
        l4_logits = torch.flatten(self.gap(l4_map), start_dim=1)  # (b, num_classes)

        logits_collect = [l1_logits, l2_logits, l3_logits, l4_logits]  # 4 x (b, num_classes)

        if return_maps:
            # Resize to input resolution (for CAMs or external models)
            re_l1 = F.interpolate(l1_map, size=(H, W), mode='bilinear', align_corners=False).detach()  # (b, num_classes, 224, 224)
            re_l2 = F.interpolate(l2_map, size=(H, W), mode='bilinear', align_corners=False).detach()  # (b, num_classes, 224, 224)
            re_l3 = F.interpolate(l3_map, size=(H, W), mode='bilinear', align_corners=False).detach()  # (b, num_classes, 224, 224)
            re_l4 = F.interpolate(l4_map, size=(H, W), mode='bilinear', align_corners=False).detach()  # (b, num_classes, 224, 224)
            map_collect = [re_l1, re_l2, re_l3, re_l4]  # 4 x (b, num_classes, 224, 224)
        else:
            map_collect = None

        if return_l1_features:
            return logits_collect, map_collect, l1  # Return l1 features (b, 64, 56, 56) for alignment
        
        return logits_collect, map_collect

    def normalize(self, tensor):
        a1, a2, a3, a4 = tensor.size()
        tensor = tensor.view(a1, a2, -1)
        min_val = tensor.min(dim=2, keepdim=True)[0]
        max_val = tensor.max(dim=2, keepdim=True)[0]
        norm = (tensor - min_val) / (max_val - min_val + 1e-5)
        return norm.view(a1, a2, a3, a4)

    def load_pretrain_weight(self, pretrain_path):
        print("Model restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)["model_state_dict"]
        state_dict_init = self.state_dict()

        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            new_state_dict[k_0] = v

        self.load_state_dict(new_state_dict, strict=False)

    def load_encoder_pretrain_weight(self, pretrain_path):
        print("Encoder restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)
        state_dict_init = self.state_dict()

        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            if k.startswith("f."):
                new_state_dict[k_0] = v

        self.load_state_dict(new_state_dict, strict=False)


class Res_Scoring(nn.Module):
    """Attention-based aggregation network for multi-scale CAMs.
    
    Inputs:
        - input: (b, 3, 224, 224)
        - map_collect: List of 4 tensors, each (b, num_classes, 224, 224)
    
    Outputs:
        - average_map: (b, 1, num_classes, 224, 224)
        - foreground: (b, num_classes, 50176) where 50176 = 224*224
        - background: (b, num_classes, 50176)
        - final_map: (b, num_classes, 224, 224)
    """
    def __init__(self, pretrain_path=None, use_unet=True, spatial_normalize=False):
        super(Res_Scoring, self).__init__()
        
        # When True, normalize over spatial h*w (dim=3) instead of num_classes (dim=2)
        # This prevents one class from suppressing the other in multiclass setting
        self.spatial_normalize = spatial_normalize
        
        # Convert 3 channel image to 1 channel but keep the same size
        self.proj = nn.Conv2d(3, 1, 1)  # (b, 3, 224, 224) -> (b, 1, 224, 224)
        self.in3 = nn.InstanceNorm2d(3, affine=True)

        # 3D attention network: (b, 4, num_classes, 224, 224) -> (b, 4, num_classes, 224, 224)
        self.att = nn.Sequential(
            nn.Conv3d(4, 32, 3, padding=1, bias=False),  # padding=1 to keep the same size
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 4, 3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU()   
        )

    def forward(self, input, map_collect, return_map_att=False, moon=False):
        # input: (b, 3, 224, 224)
        # map_collect: List of 4 x (b, num_classes, 224, 224)
        
        input = self.in3(input)  # (b, 3, 224, 224)
        
        # Stack input CAMs: [batch, 4, num_classes, h, w]
        mask = torch.stack(map_collect, dim=1)  # (b, 4, num_classes, 224, 224)

        # Normalize CAMs across spatial dimensions
        norm_mask = self.normalize(mask)  # (b, 4, num_classes, 224, 224)
        
        # Convert input RGB image to 1 channel features
        input_gray = self.proj(input)  # (b, 1, 224, 224)
        
        # Multiply grayscale input with normalized CAMs
        masked_input = input_gray.unsqueeze(1) * norm_mask  # (b, 1, 1, 224, 224) * (b, 4, num_classes, 224, 224) = (b, 4, num_classes, 224, 224)
        
        # Pass through 3D attention/aggregation module
        map_att = self.att(masked_input)  # (b, 4, num_classes, 224, 224)
        map_att_for_alignment = map_att.mean(dim=2) 
        if moon:
            return map_att_for_alignment
        # Permute for softmax across layers: [batch, 4, num_classes, h, w] -> [batch, num_classes, 4, h, w]
        map_att_permuted = map_att.permute(0, 2, 1, 3, 4)  # (b, num_classes, 4, 224, 224)
        map_weight = F.softmax(map_att_permuted, dim=2)  # (b, num_classes, 4, 224, 224) - softmax over 4 layers
        map_weight = map_weight.permute(0, 2, 1, 3, 4)  # (b, 4, num_classes, 224, 224)

        # Weighted sum of maps using attention weights
        final_map = torch.sum(mask * map_weight, dim=1)  # (b, num_classes, 224, 224)

        # Compute foreground and background features
        foreground = input_gray * final_map  # (b, 1, 224, 224) * (b, num_classes, 224, 224) = broadcast to (b, num_classes, 224, 224)
        foreground = torch.flatten(foreground, start_dim=2)  # (b, num_classes, 50176) where 50176 = 224*224
        background = input_gray * (1 - final_map)  # (b, num_classes, 224, 224)
        background = torch.flatten(background, start_dim=2)  # (b, num_classes, 50176)
        
        # Average map: combine all 5 maps (4 original + 1 final)
        map_collect.append(final_map)  # List of 5 x (b, num_classes, 224, 224)
        all_map = torch.stack(map_collect, dim=1)  # (b, 5, num_classes, 224, 224)
        average_map = torch.mean(all_map, dim=1, keepdim=True)  # (b, 1, num_classes, 224, 224)
        
        if return_map_att:
            # For alignment: averaged over num_classes -> (b, 4, 224, 224)
            return average_map, foreground, background, final_map, map_att_for_alignment
        
        return average_map, foreground, background, final_map

    def normalize(self, tensor):
        a1, a2, a3, a4, a5= tensor.size()
        tensor = tensor.view(a1, a2, a3, -1)
        if self.spatial_normalize:
            # Normalize over spatial dim (h*w) per class independently
            # This prevents one class from suppressing the other
            tensor_min = (tensor.min(3, keepdim=True)[0])
            tensor_max = (tensor.max(3, keepdim=True)[0])
        else:
            # Original: normalize over num_classes dim
            tensor_min = (tensor.min(2, keepdim=True)[0])
            tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4, a5)
        return tensor

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)["model_state_dict"]
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print(" res_score Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)["model_state_dict"]
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")



# Simple model which takes in_channels and output 3 channels and keep the height and width the same
class SimpleConvModel(nn.Module):
    """Simple convolutional model for channel conversion.
    
    Input: (b, in_channels, h, w)
    Output: (b, 3, h, w)
    """
    def __init__(self, in_channels):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (b, in_channels, h, w) -> (b, 3, h, w)
        return self.conv(x)
    

# Simple UNet: one downsampling then upsampling, output same size as input
class SimpleUNet(nn.Module):
    """Simple U-Net style architecture with one encoder-decoder stage.
    
    For input (b, in_channels, 224, 224):
        - down: (b, in_channels, 224, 224) -> (b, 64, 224, 224)
        - pool: (b, 64, 224, 224) -> (b, 64, 112, 112)
        - up: (b, 64, 112, 112) -> (b, 64, 224, 224)
        - out_conv: (b, 64, 224, 224) -> (b, 3, 224, 224)
    
    Output: (b, 3, 224, 224)
    """
    def __init__(self, in_channels):
        super(SimpleUNet, self).__init__()
        self.down = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # maintains spatial size
        self.pool = nn.MaxPool2d(2)  # halves spatial dimensions
        self.up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # doubles spatial dimensions
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # maintains spatial size
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(64)
        self.in3 = nn.InstanceNorm2d(3)
            
    def forward(self, x):
        # x: (b, in_channels, 224, 224)
        x1 = F.relu(self.in1(self.down(x)))  # (b, 64, 224, 224)
        x2 = self.pool(x1)  # (b, 64, 112, 112)
        x3 = F.relu(self.in2(self.up(x2)))  # (b, 64, 224, 224)
        x4 = self.out_conv(x3)  # (b, 3, 224, 224)
        return x4


class DoubleConv3d(nn.Module):
    """(conv3d -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class TwoLayerUNet3D(nn.Module):
    """
    2-level U-Net with a bottleneck.
    Input:  (b, in_channels=4, D, 224, 224)
    Output: (b, out_channels=4, D, 224, 224)
    """
    def __init__(self, in_channels=4, out_channels=4, base_filters=32):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = DoubleConv3d(in_channels, f)      # -> (b, f, D, 224, 224)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))# -> (b, f, D, 112, 112)

        self.enc2 = DoubleConv3d(f, f*2)              # -> (b, 2f, D, 112, 112)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))# -> (b, 2f, D, 56, 56)

        # Bottleneck
        self.bottleneck = DoubleConv3d(f*2, f*4)      # -> (b, 4f, D, 56, 56)

        # Decoder
        # Use ConvTranspose3d with kernel (1,2,2) to upsample spatial dims only
        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = DoubleConv3d(f*4, f*2)            # concat -> channels 4f -> 2f

        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = DoubleConv3d(f*2, f)              # concat -> channels 2f -> f

        # Final conv to produce desired number of channels
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)           # upsample spatially: (D stays same)
        # if needed, pad/crop u2 to match e2 spatial dims (should match for even dims)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out


class Si(nn.Module):
    """
    2-level U-Net with a bottleneck.
    Input:  (b, in_channels=4, D, 224, 224)
    Output: (b, out_channels=4, D, 224, 224)
    """
    def __init__(self, in_channels=4, out_channels=4, base_filters=32):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = DoubleConv3d(in_channels, f)      # -> (b, f, D, 224, 224)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))# -> (b, f, D, 112, 112)

        self.enc2 = DoubleConv3d(f, f*2)              # -> (b, 2f, D, 112, 112)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))# -> (b, 2f, D, 56, 56)

        # Bottleneck
        self.bottleneck = DoubleConv3d(f*2, f*4)      # -> (b, 4f, D, 56, 56)

        # Decoder
        # Use ConvTranspose3d with kernel (1,2,2) to upsample spatial dims only
        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = DoubleConv3d(f*4, f*2)            # concat -> channels 4f -> 2f

        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = DoubleConv3d(f*2, f)              # concat -> channels 2f -> f

        # Final conv to produce desired number of channels
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)           # upsample spatially: (D stays same)
        # if needed, pad/crop u2 to match e2 spatial dims (should match for even dims)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out
