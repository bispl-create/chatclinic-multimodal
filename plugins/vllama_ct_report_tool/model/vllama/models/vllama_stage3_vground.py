import logging
import random

import os
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
#from torch.profiler import profile, record_function, ProfilerActivity
from vllama.common.registry import registry
from vllama.models.blip2 import Blip2Base, disabled_train
from vllama.models.modeling_llama import LlamaForCausalLM
from vllama.models.perceiver import PerceiverResampler
from transformers import LlamaTokenizer, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms as pth_transforms
import vllama.models.vision_transformer as vits
import pdb
import kornia
import torch.distributed as dist

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def masked_mae_loss(original, reconstructed, masks):
        # Apply mask to the outputs to focus the loss calculation on masked elements
        loss = torch.abs(original - reconstructed) * masks
        return loss.sum() / masks.sum()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


###Extracting certain head attentions ONLY###

def forward_hook(module, input, output):
    module.used = True

def hook_function(module, grad_in, grad_out):
    # grad_in is a tuple containing the gradients coming into the layer
    # grad_out is a tuple containing the gradients going out of the layer
    print(f"Grad in: {grad_in[0].norm()}, Grad out: {grad_out[0].norm()}")

@registry.register_model("vllama_stage3_vground")
class vllamastage3vground(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/vllama.yaml",
    }

    def __init__(
        self,
        vit_model='ViT-B/32',
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=True,
        vit_precision="fp16",
        freeze_vit=False,
        freeze_qformer=False,
        freeze_zformer=False,
        freeze_perceiver=False,
        num_query_token=16,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=64,
        end_sym='\n',
        low_resource=True,  # use 8 bit and put vit in cpu =
        device_8bit=2,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vit_path = "/scratch/slurm-user3/changsun/dino/checkpoint/rectal_MRI_all_sorted_imgnet_pretrained/checkpoint0095.pth",
        lora_r = 0,
        lora_target_modules=["q_proj","v_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        temp=0,
        alpha=0,
        momentum=0,
        queue_size=500,
        z_path=None,
        embed_dim=768,
        heads_to_use=[0,4,10],
        depth=6,
        big_bird=True,
        evaluate=False,
        mask_in_chans = 256
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.evaluate = evaluate
        self.heads_to_use = heads_to_use
        self.mask_in_chans = mask_in_chans

        if self.evaluate:
            device = f'cuda:{device_8bit}'
            llm_device = device_8bit
        else:
            device = f'cuda:{dist.get_rank()}'
            llm_device = dist.get_rank()

        print('Loading VIT')

        ###FOR DINO
        #self.raw_attention_maps = []
        self.visual_encoder = self.init_DINO_encoder(vit_model, vit_path[0], 16, self.evaluate)
        self.visual_encoder.to(device)
        #self.visual_encoder_two = self.init_DINO_encoder(vit_model, vit_path[1], patch_size=16)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            # for name, param in self.visual_encoder_two.named_parameters():
            # #     param.requires_grad = False
            # for layer in self.visual_encoder.blocks:
            #     layer.attn.register_forward_hook(self.attention_hook)
            logging.info("freeze vision encoder")

        print('Loading VIT Done')

        ###USING Z-Former
        self.big_bird = big_bird
        self.z_path = z_path
        self.Zformer = self.init_Zformer(self.big_bird, self.z_path)
        self.Zformer.to(device)

        ###
        if big_bird:
            self.Zformer.pooler.bias.requires_grad = False
            self.Zformer.pooler.weight.requires_grad = False
            self.Zformer.embeddings.word_embeddings.weight.requires_grad = False
            self.Zformer.embeddings.position_embeddings.weight.requires_grad = False
            self.Zformer.embeddings.token_type_embeddings.weight.requires_grad = False

        else:
            self.Zformer.pooler.dense.bias.requires_grad = False
            self.Zformer.pooler.dense.weight.requires_grad = False
            self.Zformer.embeddings.word_embeddings.weight.requires_grad = False
            self.Zformer.embeddings.position_embeddings.weight.requires_grad = False
            self.Zformer.embeddings.token_type_embeddings.weight.requires_grad = False

        if freeze_zformer:
            for name, param in self.Zformer.named_parameters():
                param.requires_grad = False
            logging.info("freeze Zformer")

        print('Loading Perceiver')

        self.perceiver = PerceiverResampler(dim=4096, depth=depth, num_latents=num_query_token)
        self.perceiver.to(device)

        if freeze_perceiver:
            for name, param in self.perceiver.named_parameters():
                param.requires_grad = False
            logging.info("freeze_perceiver")

        print('Loading LLAMA')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        #self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.pad_token = '$$' #self.llama_tokenizer.eos_token
        print('POST self.llama_tokenizer.pad_token_id', self.llama_tokenizer.pad_token_id, self.llama_tokenizer.eos_token_id, self.llama_tokenizer.pad_token, self.llama_tokenizer.eos_token, self.llama_tokenizer.padding_side)

        if self.low_resource:
            print("Low_resource activated")

            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={'': llm_device}
            )

            #self.llama_model = transformers.AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, load_in_8bit=True, )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                device_map={'': llm_device}
            )

        self.llama_model_device = self.llama_model.device

        print('llama_model device', self.llama_model_device)

        if lora_r > 0:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            self.llama_model.register_backward_hook(hook_function)
            print("LLAMA MODEL under LORA")
            self.llama_model.print_trainable_parameters()

        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('LLaMA FROZEN')
        #print(self.llama_model)
        #print(self.llama_model.model)
        self.llama_model.model.model.embed_tokens.weight.requires_grad = False
        #self.llama_model.model.model.embed_tokens.bias.requires_grad = False
        #self.llama_model.embed_tokens.weight.requires_grad = False

        print('Loading LLAMA Done')
        self.perc_proj_chz = nn.Linear(768, 4096).to(self.llama_model_device)
        self.llama_proj_chz = nn.Linear(4096, self.llama_model.config.hidden_size).to(self.llama_model_device)
        #self.llama_proj_chz.register_backward_hook(hook_function)
        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv2d(1, self.mask_in_chans // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(self.mask_in_chans // 4),
        #     nn.GeLU(),
        #     nn.MaxPool2d((2,2))
        #     nn.Conv2d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2),
        #     LayerNorm2d(self.mask_in_chans),
        #     nn.GeLU(),
        #     nn.MaxPool2d((2,2))
        #     nn.Conv2d(self.mask_in_chans, self.llama_model.config.hidden_size, kernel_size=1),
        # )
        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv3d(1, self.mask_in_chans // 4, kernel_size=2, stride=2),
        #     nn.LayerNorm([self.mask_in_chans // 4, 60, 188, 188]),  # Adjust the LayerNorm to match output dimensions
        #     nn.GELU(),
        #     nn.MaxPool3d((4, 2, 2)),
        #     nn.Conv3d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2),
        #     nn.LayerNorm([self.mask_in_chans, 28, 93, 93]),  # Adjust LayerNorm dimensions again
        #     nn.GELU(),
        #     nn.MaxPool3d((4, 2, 2)),
        #     nn.Conv3d(self.mask_in_chans, self.llama_model.config.hidden_size, kernel_size=1)
        # ).to(self.llama_model_device)

        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv3d(1, self.mask_in_chans // 4, kernel_size=4, stride=2),
        #     nn.LayerNorm([self.mask_in_chans // 4, 59, 187, 187]),  # LayerNorm to match output dimensions
        #     nn.GELU(),
        #     nn.MaxPool3d((4, 4, 4)),  # Changed pooling size to (4, 4, 4)
        #     nn.Conv3d(self.mask_in_chans // 4, self.mask_in_chans // 2, kernel_size=4, stride=2),
        #     nn.LayerNorm([self.mask_in_chans // 2, 6, 22, 22]),  # Adjusted LayerNorm dimensions
        #     nn.GELU(),
        #     nn.MaxPool3d((4, 4, 4)),  # Changed pooling size to (4, 4, 4)
        #     nn.Conv3d(self.mask_in_chans // 2, self.mask_in_chans, kernel_size=1)
        # ).to(self.llama_model_device)

        #self.dim_reduction = nn.Linear(self.mask_in_chans*1*5*5, 4096).to(self.llama_model_device)
        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv3d(1, self.mask_in_chans // 4, kernel_size=(4, 4, 4), stride=(2, 2, 2)),  # Reduce depth & spatial dimensions
        #     nn.LayerNorm([self.mask_in_chans // 4, 59, 187, 187]),  # Adjusted LayerNorm dimensions
        #     nn.GELU(),
        #     nn.MaxPool3d((2, 4, 4)),  # Reduce depth and spatial dimensions further
        #     nn.Conv3d(self.mask_in_chans // 4, self.mask_in_chans // 2, kernel_size=(2, 4, 4), stride=(1, 2, 2)),  # Focus on spatial reduction
        #     nn.LayerNorm([self.mask_in_chans // 2, 28, 22, 22]),  # Adjusted LayerNorm dimensions
        #     nn.GELU(),
        #     nn.MaxPool3d((2, 4, 4)),  # Pooling that targets spatial dimensions reduction
        #     nn.Conv3d(self.mask_in_chans // 2, self.mask_in_chans, kernel_size=(1, 1, 1)),  # To achieve desired channel output
        #     nn.AdaptiveAvgPool3d((1, 4, 4))  # Adaptive pooling to achieve (1, 4, 4) shape
        # ).to(self.llama_model_device)

        # self.dim_reduction = nn.Linear(self.mask_in_chans * 4 * 4, 4096).to(self.llama_model_device)

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.prompt_dict = {}
        # self.prompt_dict["brainMRI"] = []
        # self.prompt_dict["rectalMRI"] = []
        #self.prompt_dict["CTRATE"] = []
        #self.prompt_dict["CTRATEImp"] = []
        self.prompt_dict["CTRATE"] = []
        for elem in self.prompt_list:
            # split_elem = elem.split(" ")
            # if "normal" in split_elem:
            self.prompt_dict["CTRATE"].append(elem)
            # else:
            #     self.prompt_dict["CTRATEImp"].append(elem)
            # elif "brain" in split_elem:
            #     self.prompt_dict["brainMRI"].append(elem)
            # elif "rectal" in split_elem:
            #     self.prompt_dict["rectalMRI"].append(elem)
        #vqa_prompt = '<Img><ImageHere></Img><Mask><MaskHere></Mask>'
        vqa_nomask_prompt = '<Img><ImageHere></Img>'
        vqa_prompt = '<Img><ImageHere></Img><Mask><MaskHere></Mask>'
        self.vqa_nomask_template = prompt_template.format(vqa_nomask_prompt)
        self.vqa_template = prompt_template.format(vqa_prompt)



    def mask_embeddings(self, image_embeds, mask_prob=0.3):
        batch_size, seq_length, embed_dim = image_embeds.shape
        mask = torch.rand(batch_size, seq_length, 1) < mask_prob  # Creating a mask for elements
        mask = mask.to(self.llama_model_device)
        original_embeds = image_embeds.clone()  # Copy to calculate loss later
        image_embeds = image_embeds.masked_fill(mask, 0)  # Fill masked positions with 0
        return image_embeds, original_embeds, mask

    def attention_hook(self, module, input, output):
        # Store the raw attention maps from specified heads only
        attention = output[1][:, self.heads_to_use, :, :]  # Select only specified heads
        self.raw_attention_maps.append(attention)

    def extract_embeddings_from_heads(self):
        # Initialize list to store the processed embeddings from each layer
        selected_heads_embeddings = []
        for attention in self.raw_attention_maps:
            # Calculate the mean attention over the specified heads, focusing on the CLS token
            cls_attention = attention[:, :, 0, :]  # CLS token is at position 0
            selected_heads_embeddings.append(cls_attention.mean(dim=1))  # Mean over selected heads across all layers

        # Optionally, you can process these embeddings further here
        return torch.stack(selected_heads_embeddings).mean(dim=0)

    def vit_to_cpu(self):
        #self.ln_vision.to("cpu")
        #self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
        # self.visual_encoder_two.to("cpu")
        # self.visual_encoder_two.float()
        #self.Qformer.to("cpu")
        #self.visual_encoder.float()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.queue_size % batch_size == 0  # for simplicity
        #print('image_feats.shape', image_feats.shape)
        #print('text_feats.shape', text_feats.shape)

        #print('batch_size', batch_size)
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def interpolate_pos_encoding(self, ds_embed, x):
        npatch = x.shape[1]
        N = ds_embed.shape[1]
        patch_pos_embed = ds_embed[:, :]

        bs, ds, p, dim = patch_pos_embed.size()
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim, ds)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=npatch / N, mode='linear')
        patch_pos_embed = patch_pos_embed.view(bs, p, dim, -1).permute(0, 3, 1, 2)

        return patch_pos_embed

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            #print(model_pair)
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                    #sum_param = sum(p.numel() for p in param)
                    #sum_param_m = sum(p.numel() for p in param_m)
                    #print(type(model_pair[0]).__name__, sum_param)
                    #print(type(model_pair[1]).__name__, sum_param_m)



    def encode_img(self, image, attn_mask):

        #with self.maybe_autocast():

        slice_embeds_list = []

        for idx in range(image.shape[1]):

            s_image = image[:,idx,:,:,:]
            #print('slice_image before min max', torch.max(s_image), torch.min(s_image))
            slice_image = self.transform(image[:,idx,:,:,:])
            #print('slice_image after transform', torch.max(slice_image), torch.min(slice_image))
            slice_embeds = self.visual_encoder(slice_image)#.to(self.llama_model_device)
            slice_embeds = slice_embeds.unsqueeze(dim=1)
            slice_embeds_list.append(slice_embeds)

        image_embeds = torch.cat(slice_embeds_list, dim=1)

        z_output = self.Zformer(inputs_embeds=image_embeds, attention_mask=attn_mask)

        recon_embeds = self.perc_proj_chz(z_output.last_hidden_state[:,:,:])

        query_embeds = self.perceiver(recon_embeds)


        query_embeds = query_embeds.squeeze(1)
        inputs_llama = self.llama_proj_chz(query_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.llama_model_device)

        return inputs_llama, atts_llama


    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def prompt_wrap_comb(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before_imp, p_after_imp = prompt.split('<Impression>')

            p_before_img, p_after_img = p_before_imp.split('<ImageHere>')

            p_before_img_tokens = self.llama_tokenizer(
                p_before_img, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_img_tokens = self.llama_tokenizer(
                p_after_img, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_imp_tokens = self.llama_tokenizer(
                p_after_imp, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

            p_before_img_embeds = self.llama_model.model.model.embed_tokens(p_before_img_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_img_embeds = self.llama_model.model.model.embed_tokens(p_after_img_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_imp_embeds = self.llama_model.model.model.embed_tokens(p_after_imp_tokens.input_ids).expand(batch_size, -1, -1)

            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_img_embeds, img_embeds, p_after_img_embeds], dim=1)

            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            atts_after_imp = atts_img[:, :1].expand(-1, p_after_imp_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img, p_after_imp_embeds, atts_after_imp
        else:
            return img_embeds, atts_img

    def prompt_vqa_wrap(self, img_embeds, mask_embeds, atts_img, prompt):
        p_after_embeds = torch.tensor([]).to(img_embeds.device)
        p_after_mask_embeds = torch.tensor([]).to(img_embeds.device)
        batch_size = img_embeds.shape[0]
        max_sequence_length = max(len(self.llama_tokenizer.encode(question, add_special_tokens=False)) for question in prompt)
        p_after_embeds_list = []
        p_after_mask_embeds_list = []
        for idx, question in enumerate(prompt):
            _, p_after = question.split('<ImageHere>')
            if idx == len(prompt)-1:
                p_before, p_after = question.split('<ImageHere>')
            p_before_mask, p_after_mask = p_after.split('<MaskHere>')

            encoded_question = self.llama_tokenizer.encode(
                p_after_mask,
                return_tensors="pt",
                add_special_tokens=False,
                padding='max_length',  # Pad to the maximum sequence length
                max_length=max_sequence_length  # Set the maximum sequence length
            ).to(img_embeds.device)
            p_after_mask_token_embeds = self.llama_model.model.model.embed_tokens(encoded_question)
            #print('p_after_token_embeds.shape', p_after_token_embeds.shape)
            p_after_embeds_list.append(p_after_mask_token_embeds)
            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = torch.cat(p_after_embeds_list, dim=0)
        p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=True).to(img_embeds.device)
        p_before_mask_tokens = self.llama_tokenizer(
                p_before_mask, return_tensors="pt", add_special_tokens=True).to(img_embeds.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_before_mask_embeds = self.llama_model.model.model.embed_tokens(p_before_mask_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_before_mask_embeds, mask_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])

        return wrapped_img_embeds, wrapped_atts_img


    def prompt_vqa_nomask_wrap(self, img_embeds, atts_img, prompt):
        #p_after_embeds = torch.tensor([]).to(img_embeds.device)
        batch_size = img_embeds.shape[0]
        max_sequence_length = max(len(self.llama_tokenizer.encode(question, add_special_tokens=False)) for question in prompt)
        p_after_embeds_list = []
        for idx, question in enumerate(prompt):
            _, p_after = question.split('<ImageHere>')
            if idx == len(prompt)-1:
                p_before, p_after = question.split('<ImageHere>')
            encoded_question = self.llama_tokenizer.encode(
                p_after,
                return_tensors="pt",
                add_special_tokens=False,
                padding='max_length',  # Pad to the maximum sequence length
                max_length=max_sequence_length  # Set the maximum sequence length
            ).to(img_embeds.device)
            p_after_token_embeds = self.llama_model.model.model.embed_tokens(encoded_question)
            #print('p_after_token_embeds.shape', p_after_token_embeds.shape)
            p_after_embeds_list.append(p_after_token_embeds)
            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)\
        p_after_embeds = torch.cat(p_after_embeds_list, dim=0)
        p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])

        return wrapped_img_embeds, wrapped_atts_img


    # def embed_mask(self, masks):
    #     print('masks.shape', masks.shape)
    #     mask_embedding = self.mask_downscaling(masks)
    #     bs = mask_embedding.shape[0]
    #     print('mask_embedding.shape', mask_embedding.shape)
    #     mask_embedding = mask_embedding.view(bs, 1, -1)

    #     return mask_embedding

    def forward(self, samples):

        image = samples[0].to(self.llama_model_device)
        text_s = samples[1]#.to(self.llama_model_device)
        modality = samples[2]
        #print('modality', modality)
        #test_list = [1,2,3]
        #print('list type', type(test_list))
        ##print('attn_mask', type(samples[3]), len(samples[3]))
        #print('attn_mask.shape', len(samples[3]))
        #print(samples[3])
        #attn_mask = samples[3].to(self.llama_model_device)

        bs, ds, c, h, w = image.size()
        attn_mask = torch.ones((bs, ds)).to(self.llama_model_device)
        img_embeds, atts_img = self.encode_img(image, attn_mask)

        text = " "

        # if modality[0] == 'CtRATEGroundAbnorm':

        #     seg_mask = samples[3].to(self.llama_model_device).to(torch.float16)

        #     print('text_s', text_s[0], text_s[1])
        #     #print('seg_mask.shape', seg_mask.shape, type(seg_mask))
        #     seg_embeds = self.embed_mask(seg_mask)
        #     print('seg_embeds.shape', seg_embeds.shape)
        #     ###SELECTING APPROPRIATE PROMPTS###
        #     #print('VQA Batch')
        #     question = text_s[0]
        #     #print('question', question, len(question))
        #     question_prompt = [self.vqa_template+" "+q for q in question]
        #     img_embeds, atts_img = self.prompt_vqa_wrap(img_embeds, seg_embeds, atts_img, question_prompt)
        #     text = [t + self.end_sym for t in text_s[1]]


        if modality[0][:12] == 'CtRATEGround':

            ###SELECTING APPROPRIATE PROMPTS###
            #print('VQA Batch')
            question = text_s[0]
            answer = text_s[1]
            print('q/a', question, answer)
            question_prompt = [self.vqa_nomask_template+" "+q for q in question]
            img_embeds, atts_img = self.prompt_vqa_nomask_wrap(img_embeds, atts_img, question_prompt)
            text = [t + self.end_sym for t in answer]

        # if modality[0] == 'CtRATEGroundAbnorm':

        #     ###SELECTING APPROPRIATE PROMPTS###
        #     #print('VQA Batch')
        #     question = text_s[0]
        #     seg_mask = samples[3].to(self.llama_model_device)
        #     seg_embeds = self.embed_mask(seg_mask)
        #     print('question', question, len(question))
        #     question_prompt = [self.vqa_template+" "+q for q in question]
        #     img_embeds, atts_img = self.prompt_vqa_wrap(img_embeds, seg_embeds, atts_img, question_prompt)
        #     text = [t + self.end_sym for t in text_s[1]]

        # elif modality[0] == 'CtRATEGroundLoc':

        #     ###SELECTING APPROPRIATE PROMPTS###
        #     #print('VQA Batch')
        #     question = text_s[0]
        #     seg_mask = samples[3].to(self.llama_model_device)
        #     seg_embeds = self.embed_mask(seg_mask)
        #     print('question', question, len(question))
        #     question_prompt = [self.vqa_template+" "+q for q in question]
        #     img_embeds, atts_img = self.prompt_vqa_wrap(img_embeds, seg_embeds, atts_img, question_prompt)
        #     text = [t + self.end_sym for t in text_s[1]]

        elif modality[0] == 'CTRATE':

            print('CtRATE chosen')
            prompt = random.choice(self.prompt_dict['CTRATE'])
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
            text = [t + self.end_sym for t in text_s]
            print('prompt/report:', prompt, text)


        self.llama_tokenizer.padding_side = "right"

        ## Masked Language Modeling ###
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.llama_model_device)

        #print('to_shape', to_regress_tokens.input_ids.shape)
        targets_pre = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        #print('repeat size', atts_img.shape[0]//targets_pre.shape[0])
        targets = targets_pre.repeat(atts_img.shape[0]//targets_pre.shape[0], 1)

        #print('targets new', targets.size())
        #print('atts_img shape', atts_img.shape)
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(self.llama_model_device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        print(targets.dtype, bos.dtype, to_regress_tokens.input_ids.dtype)
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        #bos_embeds = self.llama_model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)

        to_regress_embeds = to_regress_embeds.repeat(img_embeds.shape[0]//to_regress_embeds.shape[0], 1, 1)

        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)

        to_regress_tokens.attention_mask = to_regress_tokens.attention_mask.repeat(atts_img.shape[0]//to_regress_tokens.attention_mask.shape[0], 1)

        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        llm_loss = outputs.loss

        return {"loss": llm_loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", 'vit_base') #"ViT-B/32" #"eva_clip_g"
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token", 32)
        llama_model = cfg.get("llama_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", True)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_zformer = cfg.get("freeze_zformer", True)
        freeze_perceiver = cfg.get("freeze_perceiver", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 2)
        vit_path = cfg.get("vit_path")
        lora_r = cfg.get("lora_r")
        lora_target_modules = cfg.get("lora_target_modules")
        lora_alpha = cfg.get("lora_alpha")
        lora_dropout = cfg.get("lora_dropout")
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 64)
        end_sym = cfg.get("end_sym", '\n')
        temp = cfg.get("temp", 0.07)
        alpha = cfg.get("alpha", 0.04)
        momentum = cfg.get("momentum", 0.995)
        queue_size = cfg.get("queue_size", 256)
        z_path = cfg.get("z_path", None)
        heads_to_use = cfg.get("heads_to_use", [0,4,10])
        depth = cfg.get("depth", 6)
        big_bird = cfg.get("big_bird", True)
        evaluate = cfg.get("evaluate", False)
        mask_in_chans = cfg.get("mask_in_chans", 256)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            freeze_zformer=freeze_zformer,
            freeze_perceiver=freeze_perceiver,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            vit_path=vit_path,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            temp=temp,
            alpha=alpha,
            momentum=momentum,
            queue_size=queue_size,
            depth=depth,
            big_bird=big_bird,
            evaluate=evaluate,
            mask_in_chans = mask_in_chans
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            init_state_dict = model.state_dict()

            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            for key, loaded_tensor in model.state_dict().items():
                initial_tensor = init_state_dict[key]
                if not torch.equal(loaded_tensor, initial_tensor):
                    print(f"Layer '{key}' has been updated with checkpoint weights.")
            #print('Message', msg)

        return model
