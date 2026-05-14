import logging

import os
import torch
from torch.cuda.amp import autocast as autocast

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
#from torch.profiler import profile, record_function, ProfilerActivity

from .perceiver import PerceiverResampler
from transformers import AutoTokenizer, OPTForCausalLM
from torchvision import transforms as pth_transforms
from . import vision_transformer as vits

from transformers import BertConfig, BertModel, BigBirdConfig, BigBirdModel
from torch.profiler import profile, record_function, ProfilerActivity
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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

class vllamastage2(nn.Module):
    """
    BLIP2 GPT-LLAMA model.
    """

    def __init__(
        self,
        vit_model='vit_base',
        freeze_vit=True,
        freeze_zformer=True,
        freeze_perceiver=True,
        num_query_token=32,
        llama_model='facebook/opt-1.3b',
        low_resource=True,  # use 8 bit and put vit in cpu =
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vit_path = '',
        lora_r = 64,
        lora_target_modules=["q_proj","k_proj", "v_proj", "o_proj"],
        lora_alpha = 128,
        lora_dropout = 0.05,
        z_path=None,
        heads_to_use=[0,4,10],
        depth=6,
        big_bird=False,
        evaluate=True,
        ve_type=None
    ):
        super().__init__()

        self.low_resource = low_resource
        self.transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.evaluate = evaluate
        self.heads_to_use = heads_to_use
        self.ve_type = ve_type

        if self.evaluate:
            device = f'cuda:{device_8bit}'
            llm_device = device_8bit
        else:
            device = f'cuda:{dist.get_rank()}'
            llm_device = dist.get_rank()

        print('Loading VIT')

        vision_encoder = None
        #print('vits.__dict__.keys()', vits.__dict__.keys())
        if vit_model in vits.__dict__.keys():
            vision_encoder = vits.__dict__[vit_model](patch_size=16)
        if vit_path != None:
            #vision_encoder.load_state_dict(torch.load(vit_path))
            if evaluate:
                state_dict = torch.load(vit_path, weights_only=False)
            else:
                state_dict = torch.load(vit_path, map_location=f'cuda:{dist.get_rank()}', weights_only=False)
            if isinstance(state_dict, dict):
                state_dict = (
                    state_dict.get("teacher")
                    or state_dict.get("student")
                    or state_dict.get("state_dict")
                    or state_dict.get("model")
                    or state_dict
                )
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            vision_encoder.load_state_dict(state_dict, strict=False)

        self.visual_encoder = vision_encoder
        self.visual_encoder.to(device)

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

        Zformer = None
        if big_bird:
            Zformer = BigBirdModel(BigBirdConfig(hidden_size=768, block_size=16, num_random_blocks=3, max_position_embeddings=512))
                #Zformer = BigBirdModel(BigBirdConfig(hidden_size=768, attention_type='original_full')
        else:
            Zformer = BertModel(BertConfig(hidden_size=768, num_attention_heads=8, max_position_embeddings=512))

        self.Zformer = Zformer
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

        self.perceiver = PerceiverResampler(dim=2048, depth=depth, num_latents=num_query_token)
        self.perceiver.to(device)

        if freeze_perceiver:
            for name, param in self.perceiver.named_parameters():
                param.requires_grad = False
            logging.info("freeze_perceiver")

        print('Loading LLAMA')

        self.llama_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

        if self.low_resource:
            print("Low_resource activated")
            self.llama_model = OPTForCausalLM.from_pretrained(
                llama_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={'': llm_device}
            )

        else:
            self.llama_model = OPTForCausalLM.from_pretrained(
                llama_model,
                device_map={'': llm_device}
            )

        self.llama_model_device = self.llama_model.device

        print('llama_model device', self.llama_model_device)
        if lora_r > 0:
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            print("LLAMA MODEL under LORA")
            self.llama_model.print_trainable_parameters()
        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('LLaMA FROZEN')

        self.llama_model.model.model.decoder.embed_tokens.weight.requires_grad = False

        print('Loading LLAMA Done')
        #print('llama_model config hidden size', self.llama_model.config.hidden_size)
        self.perc_proj_chz = nn.Linear(768, 2048).to(self.llama_model_device)
        self.llama_proj_chz = nn.Linear(2048, self.llama_model.config.hidden_size).to(self.llama_model_device)

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
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def interpolate_pos_encoding(self, ds_embed, x):
        npatch = x.shape[1]
        N = ds_embed.shape[1]
        patch_pos_embed = ds_embed[:, :]

        bs, ds, p, dim = patch_pos_embed.size()
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim, ds)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, scale_factor=npatch / N, mode='linear')
        patch_pos_embed = patch_pos_embed.view(bs, p, dim, -1).permute(0, 3, 1, 2)

        return patch_pos_embed

    def encode_img(self, image, attn_mask):

        slice_embeds_list = []

        for idx in range(image.shape[1]):

            slice_image = image[:,idx,:,:,:]
            min_slice = slice_image.min()
            max_slice = slice_image.max()

            if max_slice != min_slice:
                # Normalize the slice image
                slice_image = self.transform(slice_image)
            else:
                print(min_slice, max_slice, 'slice_image is not normalized')

            slice_embeds = self.visual_encoder(slice_image) #.to(self.llama_model_device)
            slice_embeds = slice_embeds.unsqueeze(dim=1).detach()
            slice_embeds_list.append(slice_embeds)
            del slice_image

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
            p_before_embeds = self.llama_model.model.model.decoder.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.model.decoder.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):

        image = samples[0].to(self.llama_model_device)
        attn_mask = samples[1].to(self.llama_model_device)
        img_embeds, atts_img = self.encode_img(image, attn_mask)

        return img_embeds, atts_img
