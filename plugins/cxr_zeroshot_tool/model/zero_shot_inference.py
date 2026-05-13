import numpy as np
import h5py
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import torch.nn.functional as F
import clip
import time
from .model import CLIP


class CXRTestDataset(data.Dataset):
    """
    Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)

        if img.ndim == 3 and img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float() # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample

def load_clip(model_path, pretrained=False, context_length=77, device=None): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    if device is None:
        device = torch.device("cpu")

    if pretrained is False: 
        # use new model params
        params = {
            'embed_dim':768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length, 
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }

        model = CLIP(**params)
    else: 
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 

    if model_path is not None: 
        # print(f"[*] Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_ema' in checkpoint:
            state_dict = checkpoint['model_ema']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if name.startswith("clip."):
                name = name.replace("clip.", "")
            if "classifier" in name:
                continue
            new_state_dict[name] = v

        try:
            model.load_state_dict(new_state_dict, strict=True)
            # print("[*] Weights Loaded")
        except RuntimeError:
            print("    [!] Strict Load Failed. Retrying with strict=False...")
            model.load_state_dict(new_state_dict, strict=False)

    return model.to(device).eval()

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
    device: torch.device = torch.device("cpu")
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length,
        device=device
    ).to(device).eval()
    
    # load data
    transformations = [
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False,pin_memory=(device.type=='cuda'))
    
    return model, loader

def run_ensemble_eval(model, loader, prompts_dict, context_length=77):
    """
    FUNCTION: run_ensemble_eval
    --------------------------------------
    Replaces run_softmax_eval. Instead of using a single template pair, 
    it takes a dictionary of multiple positive/negative prompts per class,
    averages their embeddings (ensemble), and calculates probabilities.
    
    args:
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader
        * prompts_dict - dict, structure: 
            {
                "Class_Name": {
                    "positive": ["prompt1", "prompt2"...], 
                    "negative": ["neg_prompt1", "neg_prompt2"...]
                }, ...
            }
        * context_length - int, max tokens
        
    Returns:
        * y_pred - numpy array of shape (num_samples, num_classes)
                   containing the probability of the positive class.
    """
    device = next(model.parameters()).device
    model.eval()

    # Pre-compute Text Embeddings
    # print("[*] Computing Ensemble Text Embeddings...")
    class_weights = {}

    # Assuming prompts_dict keys are the labels
    target_labels = list(prompts_dict.keys())

    with torch.no_grad():
        for label in target_labels:
            texts = prompts_dict[label]
            
            # Encode Positive Prompts
            pos_text_tokens = clip.tokenize(texts['positive'], context_length=context_length).to(device)
            pos_feats = model.encode_text(pos_text_tokens)
            pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True) # Normalize individual
            pos_proto = pos_feats.mean(dim=0) # Average
            pos_proto = pos_proto / pos_proto.norm() # Normalize prototype

            # Encode Negative Prompts
            neg_text_tokens = clip.tokenize(texts['negative'], context_length=context_length).to(device)
            neg_feats = model.encode_text(neg_text_tokens)
            neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
            neg_proto = neg_feats.mean(dim=0)
            neg_proto = neg_proto / neg_proto.norm()

            # Stack [Positive, Negative] -> Shape [2, 768]
            class_weights[label] = torch.stack([pos_proto, neg_proto])

    # Image Inference Loop
    y_pred_all = []

    # Inference time
    total_time = 0
    num_images = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Ensemble Inference")):
            images = data['img'].to(device)

            torch.cuda.synchronize()
            start = time.time()
        
            # Encode Original Image
            image_features_orig = model.encode_image(images)
            image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)

            # Encode Zoomed Image (Center Crop)
            _, _, h, w = images.shape
            zoom_ratio = 0.85

            ch, cw = int(h * zoom_ratio), int(w * zoom_ratio)
            sy, sx = (h - ch) // 2, (w - cw) // 2

            images_crop = images[:, :, sy:sy+ch, sx:sx+cw]
            images_zoom = F.interpolate(
                images_crop,
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )

            image_features_zoom = model.encode_image(images_zoom)
            image_features_zoom = image_features_zoom / image_features_zoom.norm(dim=-1, keepdim=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            total_time += (end - start)
            num_images += images.shape[0]

            batch_preds = []

            # Calculate probability for each class independently
            for label in target_labels:
                weights = class_weights[label] # [2, 768]
                                                
                if label == "Scoliosis":
                    img_feat = image_features_orig

                else:
                    img_feat = (image_features_orig + image_features_zoom) / 2.0
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                img_feat = image_features_orig
                

                logits = img_feat @ weights.t()
                
                # Softmax across the Pos/Neg dimension
                probs = F.softmax(logits, dim=-1) 
                
                # Take only the Positive probability (Column 0)
                pos_prob = probs[:, 0].cpu().numpy()
                batch_preds.append(pos_prob)

            batch_preds = np.stack(batch_preds, axis=1)
            y_pred_all.append(batch_preds)
        
        avg_time = total_time / num_images
        # print(f"Average inference time per image: {avg_time:.6f} sec")

    # Concatenate all batches
    y_pred_final = np.concatenate(y_pred_all, axis=0)
    
    return y_pred_final