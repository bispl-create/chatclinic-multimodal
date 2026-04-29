import torch
import torch.nn as nn
import timm
import sys
import torch.nn.functional as F

# =========================================================
# PCAM Implementation
# =========================================================
class PCAMPooling(nn.Module):
    """
    A PCAM Pooling layer
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat_map, logit_map):
        # Logit clip; prevent extreme value.
        logit_map = torch.clamp(logit_map, min=-10.0, max=10.0)
        
        prob_map = torch.sigmoid(logit_map)
        
        # Safe division
        sum_prob = prob_map.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        weight_map = prob_map / (sum_prob + 1e-6)
        
        # Weight clipping
        weight_map = torch.clamp(weight_map, max=100.0)
        
        weighted_feat_map = feat_map.unsqueeze(1) * weight_map.unsqueeze(2)
        feat = weighted_feat_map.sum(dim=(3, 4), keepdim=True) 
        
        return feat, weighted_feat_map

# =========================================================
# ConvNext Backbone + PCAM Pooling layer
# =========================================================
class ConvNeXtV2_newPCAM(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', num_classes=30, drop_rate=0.5, pretrained=True):
        super(ConvNeXtV2_newPCAM, self).__init__()
        
        # print(f"[*] Initializing ConvNeXt V2 Backbone: {model_name}")
        
        # 1. Backbone (timm)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
        )
        
        # 2. ImageNet Normalization Constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.print_stats = True 
        
        # 3. Output Channel Auto-Detection
        # Put in dummy image to get output vector dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            last_feat = feats[-1] 
            self.num_features = last_feat.shape[1]
            # print(f"    -> Backbone Channels: {self.num_features}")
        
        # 4. PCAM Head Elements
        # print("[*] PCAM . new version")
        self.cam_conv = nn.Conv2d(self.num_features, num_classes, kernel_size=1)
        self.pcam_pool = PCAMPooling()
        
        self.bn = nn.BatchNorm2d(self.num_features)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cls_conv = nn.Conv2d(self.num_features, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Input Adapter
        # Pixel values -1024~1024 -> ImageNet Scale
        
        # Channel change into 3.
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # Normalize to [0, 1]
        x = (x + 1024.0) / 2048.0
        x = torch.clamp(x, 0.0, 1.0)       
        if self.print_stats:
            min_01, max_01 = x.min().item(), x.max().item()
        # Imagenet Rescale 
        x = (x - self.mean) / self.std
        
        # 2. Backbone & PCAM
        features = self.backbone(x)
        feat_map = features[-1] 

        logit_map = self.cam_conv(feat_map)
        pooled_feat, weighted_feat_map = self.pcam_pool(feat_map, logit_map)

        #class_feats [B,30,1024]
        class_feats = pooled_feat.squeeze(-1).squeeze(-1) #(B,N_class,C_channel,1,1)->(B,N_class,C_channel)
        #cls.conv.weight [30,1024] 
        classifier_weights = self.cls_conv.weight.squeeze(-1).squeeze(-1)

        logits = (class_feats * classifier_weights).sum(dim=2) #(B,N)
        
        if self.cls_conv.bias is not None:
            logits += self.cls_conv.bias
        return logits, logit_map, weighted_feat_map







# =========================================================
#  CaiT (Class-Attention in Image Transformers) Components
# =========================================================

class LayerScale(nn.Module):
    """CaiT LayerScale"""
    def __init__(self, dim, init_values=1e-4, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SelfAttentionBlock(nn.Module):
    """Stage 1: Self-Attention (Patches interact, No CLS)"""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ls1 = LayerScale(dim, init_values=init_values)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class ClassAttentionBlock(nn.Module):
    """Stage 2: Class-Attention (CLS queries Patches)"""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ls1 = LayerScale(dim, init_values=init_values)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x_cls, x_patches):
        # Concatenate for Norm (Efficient Implementation)
        u = torch.cat((x_cls, x_patches), dim=1)
        u = self.norm1(u)
        z_norm = u[:, 0:1]
        x_patches_norm = u[:, 1:]
        
        # Cross Attention: Q=CLS, K=Patches, V=Patches
        attn_out, _ = self.attn(z_norm, x_patches_norm, x_patches_norm)
        
        # Update CLS
        x_cls = x_cls + self.ls1(attn_out)
        x_cls = x_cls + self.ls2(self.mlp(self.norm2(x_cls)))
        return x_cls

class CaiTHead(nn.Module):
    def __init__(self, in_channels=768, embed_dim=256, num_classes=30, 
                 sa_depth=12, ca_depth=2, num_heads=8):
        super().__init__()
        
        # 1. Projector (ConvNeXt Feature -> ViT Dim)
        self.projector = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # 2. Positional Embedding (Fixed 16x16)
        self.num_patches = 16 * 16 
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. Stage 1: Self-Attention Layers
        self.sa_layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads) for _ in range(sa_depth)
        ])
        
        # 4. Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 5. Stage 2: Class-Attention Layers
        self.ca_layers = nn.ModuleList([
            ClassAttentionBlock(embed_dim, num_heads) for _ in range(ca_depth)
        ])
        
        # 6. Classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, 768, 16, 16]
        
        x = self.projector(x)            # [B, 256, 16, 16]
        x = x.flatten(2).transpose(1, 2) # [B, 256, 256]
        x = x + self.pos_embed
        
        # Stage 1 (SA)
        for layer in self.sa_layers:
            x = layer(x)
            
        # Stage 2 (CA)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for layer in self.ca_layers:
            cls_tokens = layer(cls_tokens, x)
            
        x = self.norm(cls_tokens)
        x = x.squeeze(1)
        return self.head(x)



# =========================================================
# ConvNeXt + CaiT
# =========================================================
class ConvNeXt_CaiT_Hybrid(nn.Module):
    """
    Pretrained ConvNeXt with PCAM, and add CaiT head for classification
    S. Park et al. (2021) Style
    """
    def __init__(self, backbone_path, num_classes=30, sa_depth=12, backbone_freeze=True, model_name='convnextv2_tiny'):
        super().__init__()

        self.backbone_freeze = backbone_freeze
        self.sa_depth = sa_depth
        
        # print(f"[*] Initializing Hybrid Model")
        
        # 1. Initialize Backbone (Uses ConvNeXtV2_PCAM defined above)
        # pretrained=False because we will load custom weights manually
        backbone_wrapper = ConvNeXtV2_newPCAM(model_name=model_name, num_classes=30, pretrained=False)
        
        # 2. Load Weights
        # print(f"    -> Loading Backbone from: {backbone_path}")
        try:
            checkpoint = torch.load(backbone_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Remove 'module.' prefix if distributed training was used
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            backbone_wrapper.load_state_dict(new_state_dict, strict=False)
            # print("    -> Backbone Weights Loaded.")
        except Exception as e:
            print(f"[!] Error loading weights: {e}")
            sys.exit(1)

        # 3. Extract Core & Freeze
        self.backbone = backbone_wrapper.backbone
        self.register_buffer('mean', backbone_wrapper.mean)
        self.register_buffer('std', backbone_wrapper.std)
        self.num_features = backbone_wrapper.num_features # 768

        del backbone_wrapper # Remove the wrapper

        for param in self.backbone.parameters():
            param.requires_grad = not self.backbone_freeze
        # print(f"    -> Backbone {'Frozen' if self.backbone_freeze else 'Unfrozen'}.")
        
        # 4. Attach CaiT Head
        # print(f"    -> Attaching CaiT Head ({self.sa_depth} SA + 2 CA)")
        self.cait_head = CaiTHead(in_channels=self.num_features, num_classes=num_classes, sa_depth=self.sa_depth)

    def forward(self, x):
        # 1. Input Adapter (Re-implemented for safety)
        # Assuming input is -1024~1024 from DataLoader
        x = (x + 1024.0) / 2048.0
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.mean) / self.std
        
        # 2. Backbone Forward (No Grad)
        if self.backbone_freeze:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        feat_map = features[-1] # [B, 768, 16, 16], CLS token output
        
        # 3. Head Forward
        logits = self.cait_head(feat_map)
        
        return logits
    


# =========================================================
# Model: Swin Transformer only
# =========================================================
class SwinMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, img_size: int = 512, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.print_stats = True

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

    def forward(self, x):
        # x: (B,1,H,W) or (B,3,H,W), value expected -1024~1024
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # -1024~1024 -> 0~1
        x = (x + 1024.0) / 2048.0
        x = torch.clamp(x, 0.0, 1.0)

        x = (x - self.mean) / self.std
        logits = self.backbone(x)  # (B, num_classes)

        if self.print_stats:
            print(f"[Swin Input] after scale: min={x.min().item():.3f}, max={x.max().item():.3f}")
            self.print_stats = False

        return logits






# =================================================
# PCAM(convnext_v2 backbone) + Swin Transformer V2 
# =================================================
class ConvFusion(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', num_classes=30, pcam_reduce_dim=24):
        super(ConvFusion, self).__init__()
        
        print(f"[*] Initializing CONVFUSION Model (PCAM + SwinV2 Two-Stream)")
        print(f"    - Branch A: PCAM with {model_name}")
        print(f"    - Branch B: Swin Transformer V2 (Pretrained)")

        # =========================================================
        # 1. Shared Input Adapter
        # =========================================================
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.print_stats = True

        # =========================================================
        # 2. Branch A: PCAM (Backbone: ConvNeXt V2)
        # =========================================================
 
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        
        # Get Backbone Channels (Base: 1024)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)

            feat_shape = self.backbone(dummy)[-1].shape 
            self.num_features = feat_shape[1]
            print(f"    -> ConvNeXt Channels: {self.num_features} (Shape: {feat_shape})")

        # PCAM Layers
        self.cam_conv = nn.Conv2d(self.num_features, num_classes, kernel_size=1)
        self.pcam_pool = PCAMPooling()
        
        self.pcam_reduce_dim = pcam_reduce_dim
        self.pcam_reduce = nn.Linear(self.num_features, self.pcam_reduce_dim)
        
        # PCAM Output Dimension = num_class * reduce_dim (30 * 24 = 720)
        self.pcam_out_dim = num_classes * self.pcam_reduce_dim 

        # =========================================================
        # 3. Branch B: Swin Transformer V2 (Global Expert)
        # =========================================================
        # ImageNet Pretrained
        self.swin_vit = timm.create_model(
            'swinv2_tiny_window16_256.ms_in1k', 
            pretrained=True, 
            num_classes=0, # No classifier head, return feature vector (1024)
            img_size=256   
        )
        self.swin_out_dim = self.swin_vit.num_features # 1024

        # =========================================================
        # 4. Fusion Head
        # =========================================================
        # Input Dim: PCAM(720) + Swin(1024) 
        final_dim = self.pcam_out_dim + self.swin_out_dim
        
        self.bn = nn.BatchNorm1d(final_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(final_dim, num_classes)
        
        # Classifier Init
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        
        # Raw Data Scaling
        x = (x + 1024.0) / 2048.0
        x = torch.clamp(x, 0.0, 1.0)
        
        if self.print_stats:
            min_01, max_01 = x.min().item(), x.max().item()
            
        # ImageNet Normalization
        x = (x - self.mean) / self.std
        
        if self.print_stats:
            print(f"\n[ConvFusion Internal] Input Rescaled: {min_01:.2f}~{max_01:.2f}")
            self.print_stats = False

        # ---------------------------------------------------------
        # Branch A: PCAM (ConvNeXt Path)
        # ---------------------------------------------------------
        # 1. Backbone Feature Extract
        features = self.backbone(x)
        feat_map = features[-1] # (B, 1024, 16, 16) at 512px

        # 2. PCAM Mechanism
        logit_map = self.cam_conv(feat_map)
        pooled_feat, _ = self.pcam_pool(feat_map, logit_map) # (B, 30, 1024, 1, 1)
        
        # 3. Reduce Dimension & Flatten
        pcam_vec = pooled_feat.squeeze(-1).squeeze(-1) # (B, 30, 1024)
        pcam_vec = self.pcam_reduce(pcam_vec)          # (B, 30, 24)
        pcam_vec_flat = pcam_vec.flatten(1)            # (B, 720)

        # ---------------------------------------------------------
        # Branch B: Swin Transformer (Independent Path)
        # ---------------------------------------------------------
        x_swin = F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=False)
        swin_vec = self.swin_vit(x_swin) # (B, 1024)

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------
        # (B, 720) + (B, 1024) -> (B, 1744)
        combined = torch.cat((pcam_vec_flat, swin_vec), dim=1)
        
        out = self.bn(combined)
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out