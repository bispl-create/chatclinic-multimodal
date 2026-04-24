# ChatClinic Segmentation Tools — Background Papers

> KAIST AI.61900 (AI for Medical Imaging and Signals, Spring 2026) — ChatClinic term project
> Topic: Segmentation · 5 tools
> Team: Chaehyeon Kim · Woojin Na · Hyunjun Park

---

## 1. Per-tool background

### 1.1 `@seg_brain` — Brain Tumor Segmentation

- **Baseline model**: SegResNet [1]
- **Model**: A ResNet-style encoder + dense decoder with an additional VAE branch that reconstructs the input image to regularize the *shared encoder* under the small BraTS dataset regime. Winner of the BraTS 2018 challenge.
- **ChatClinic usage**: 4-channel MRI (T1, T1ce, T2, FLAIR) → 3-region overlapping label (ET / TC / WT), served from the MONAI `brats_mri_segmentation` bundle.

### 1.2 `@seg_lung` — Chest X-ray Lung Segmentation

- **Baseline model**: SARNet [2] — built on a PSPNet [3] backbone
- **Model**: SARNet (Lian et al.) trains a PSPNet segmentation backbone on the ChestX-Det dataset to predict 14 thoracic structures, including left/right lung. The PSPNet backbone (Zhao et al.) uses a ResNet + dilated-convolution feature extractor followed by a 4-level pyramid pooling module (bin sizes 1×1 / 2×2 / 3×3 / 6×6) that aggregates global and multi-scale local context simultaneously.
- **ChatClinic usage**: We load the 14-class SARNet checkpoint via the **TorchXRayVision** library, then post-process the output by selecting the two lung channels and merging them into a binary lung mask.

### 1.3 `@seg_organ` — Multi-Organ CT Segmentation

- **Baseline model**: MaskSAM [4]
- **Model**: An extension of the Segment Anything Model to volumetric medical segmentation. (1) A *prompt generator* that removes the manual-prompt requirement by automatically producing auxiliary classifier tokens, binary masks, and bounding boxes; (2) a 3D depth-convolution adapter (for image embeddings) and a 3D depth-MLP adapter (for prompt embeddings) that efficiently 3D-fine-tune the 2D SAM backbone; (3) mask-classification-style multi-class prediction. State-of-the-art on AMOS · ACDC · Synapse.
- **ChatClinic usage**: We integrate the MaskSAM architecture (SAM ViT-H + prompt generator + 3D adapters) on top of the vendored **nnU-Net v2** pipeline and train it ourselves on the AMOS 2022 CT split. The resulting model outputs, in a single forward pass, the background plus all 15 abdominal organ classes of AMOS22 (spleen, R./L. kidney, gallbladder, esophagus, liver, stomach, aorta, IVC, pancreas, R./L. adrenal gland, duodenum, bladder, prostate/uterus).

### 1.4 `@seg_spleen` — Spleen CT Segmentation

- **Baseline model**: Residual U-Net [5]
- **Model**: A standard U-Net in which each convolution block is replaced by a residual unit, mitigating gradient vanishing in deep 3D networks and stabilizing sliding-window 3D inference. The original paper targeted the STACOM18 LV Quantification challenge, but its task-agnostic structure has since been reused as a baseline for single-organ abdominal segmentation.
- **ChatClinic usage**: 2-class (background / spleen) NIfTI mask, served from the MONAI `spleen_ct_segmentation` bundle.

### 1.5 `@seg_pancreas` — Pancreas + Tumor CT Segmentation

- **Baseline model**: DiNTS [6]
- **Model**: A differentiable NAS framework dedicated to 3D medical segmentation. The multi-path topology search space is converted into a sequential space via "super feature nodes" and explored with continuous relaxation; a topology-guaranteed discretization algorithm minimizes the discretization gap between the searched continuous model and the deployed discrete model. GPU memory budget is imposed as an explicit constraint. State-of-the-art on the MSD 10 tasks with only 5.8 GPU-days of search.
- **ChatClinic usage**: 3-class (background / pancreas / pancreatic tumor) NIfTI mask, served from the MONAI `pancreas_ct_dints_segmentation` bundle.

---

## 2. References

- **[1]** A. Myronenko. "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization." *BrainLes / MICCAI Workshop*, 2018. (arXiv:1810.11654)
- **[2]** J. Lian, J. Liu, S. Zhang, K. Gao, X. Liu, D. Zhang, Y. Yu. "A Structure-Aware Relation Network for Thoracic Diseases Detection and Segmentation." *IEEE Transactions on Medical Imaging*, vol. 40, pp. 2042–2052, 2021.
- **[3]** H. Zhao, J. Shi, X. Qi, X. Wang, J. Jia. "Pyramid Scene Parsing Network." *CVPR*, 2017.
- **[4]** B. Xie, H. Tang, B. Duan, D. Cai, Y. Yan, G. Agam. "MaskSAM: Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation." *ICCV*, 2025, pp. 24423–24433.
- **[5]** E. Kerfoot, J. Clough, I. Oksuz, J. Lee, A. P. King, J. A. Schnabel. "Left-Ventricle Quantification Using Residual U-Net." *STACOM 2018 Workshop @ MICCAI*, Springer LNCS 11395, 2019.
- **[6]** Y. He, D. Yang, H. Roth, C. Zhao, D. Xu. "DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation." *CVPR*, 2021.
