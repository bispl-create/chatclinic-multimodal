import os
#os.environ['LOCAL_RANK'] = "0"

import logging
import random


import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
#from torch.profiler import profile, record_function, ProfilerActivity
from vllama.common.registry import registry
from vllama.models.blip2 import Blip2Base, disabled_train
from vllama.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, BertTokenizer
from transformers import AutoTokenizer, AutoModel
import vllama.models.vision_transformer as vits
import pdb
import kornia

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#@torch.no_grad()
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

def forward_hook(module, input, output):
    module.used = True

def hook_function(module, grad_in, grad_out):
    # grad_in is a tuple containing the gradients coming into the layer
    # grad_out is a tuple containing the gradients going out of the layer
    print(f"Grad in: {grad_in[0].norm()}, Grad out: {grad_out[0].norm()}")

@registry.register_model("vllamaita_zqformer_prealign")
class vllamaItaZQformerPreAlign(Blip2Base):
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
        num_query_token=8,
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
        embed_dim=768
    ):
        super().__init__()
        print('num_query_token', num_query_token)
        self.llama_model_device = 'cuda'

        self.tokenizer = self.init_tokenizer()
        self.end_sym = end_sym
        self.low_resource = low_resource

        print('Loading VIT')

        ###FOR BLIP-2
        '''
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        '''
        ###FOR CHeXZero


        ###FOR CheXZero
        print("CHECK UPDATE")
        #self.visual_encoder = self.init_CheXzero_encoder(vit_path)
        #self.visual_encoder_m = self.init_CheXzero_encoder(vit_path)
        '''
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

            #for name, param in self.ln_vision.named_parameters():
                #param.requires_grad = False
            #self.ln_vision = self.ln_vision.eval()
            #self.ln_vision.train = disabled_train

            logging.info("freeze vision encoder")
        '''
        ###FOR DINO

        self.visual_encoder = self.init_DINO_encoder(vit_model, vit_path[0], patch_size=16)
        #self.visual_encoder_two = self.init_DINO_encoder(vit_model, vit_path[1], patch_size=16)

        self.visual_encoder.to(self.llama_model_device)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            # for name, param in self.visual_encoder_two.named_parameters():
            #     param.requires_grad = False
            logging.info("freeze vision encoder")


        print('Loading VIT Done')

        ###USING Z-Former
        self.z_path = z_path
        self.Zformer = self.init_Zformer(self.z_path)

        ###
        self.Zformer.pooler.dense.bias.requires_grad = False
        self.Zformer.pooler.dense.weight.requires_grad = False
        self.Zformer.embeddings.word_embeddings.weight.requires_grad = False

        self.Zformer.to(self.llama_model_device)
        ###USING Q-Former
        print('Loading Q-Former')
        #print('visual_encoder.num_features', self.visual_encoder.num_features)
        print('num_query_token.shape', num_query_token)
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None

        self.Qformer_m, _ = self.init_Qformer(num_query_token, 1408)
        self.Qformer_m.cls = None

        self.load_from_pretrained(url_or_filename=q_former_model)
        self.Qformer = self.Qformer.train()
        self.Qformer.bert.register_backward_hook(hook_function)

        if freeze_qformer:
            for layer in self.Qformer.bert.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")

        self.Qformer.to(self.llama_model_device)


        for name, param in self.Qformer_m.named_parameters():
                param.requires_grad = False ###Fine-tune QFormer with CheXzero vision encoder
        self.Qformer_m = self.Qformer_m.eval()
        self.Qformer_m.train = disabled_train

        print('Loading Q-Former Done')

        ###END Q-Former


        print('numfeatures: ', self.visual_encoder.num_features)
        self.qform_proj_chz = nn.Linear(self.visual_encoder.num_features, 1408).to(self.llama_model_device) #self.visual_encoder.num_features

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.alphas = alpha
        self.momentum = momentum

        ###QUEUE
        self.embed_dim = 768
        self.queue_size = queue_size
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


        self.model_pairs = [[self.Qformer, self.Qformer_m],
                            #[self.llama_proj_chz, self.llama_proj_chz_m],
                            #[self.qform_proj_chz, self.qform_proj_chz_m],
                            #[self.vision_proj, self.vision_proj_m],
                            #
                           # [self.text_proj, self.text_proj_m],
                            #[self.z_embed, self.z_embed_m],
                            # [self.zt_embed, self.zt_embed_m],
                            # [self.z_transformer, self.z_transformer_m],
                            # [self.overall_token, self.overall_token_m]
                            ]

        '''
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filtered_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filtered_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.prompt_dict = {}
        self.prompt_dict["brainMRI"] = []
        self.prompt_dict["rectalMRI"] = []
        for elem in self.prompt_list:
            split_elem = elem.split(" ")
            if "brain" in split_elem:
                self.prompt_dict["brainMRI"].append(elem)
            elif "rectal" in split_elem:
                self.prompt_dict["rectalMRI"].append(elem)
        '''
    def vit_to_cpu(self):
        #self.ln_vision.to("cpu")
        #self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
        # self.visual_encoder_two.to("cpu")
        # self.visual_encoder_two.float()
        #self.Qformer.to("cpu")
        #self.visual_encoder.float()

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

            '''
            try:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            except:
                model_pair[1].data = model_pair[1].data * self.momentum + model_pair[0].data * (1. - self.momentum)
            '''

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

    def encode_img(self, image, attn_mask, modality):
        #device = self.llama_model_device
        #print('Device', device)
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        print('modality')
        with self.maybe_autocast():

            image_embeds = torch.tensor([]).to(self.llama_model_device)
            slice_embeds = torch.tensor([]).to(self.llama_model_device)
            print('image encoding START')
            for idx in range(image.shape[1]):

                slice_image = image[:,idx,:,:,:]
                #slice_attn_mask = attn_mask[:,idx]
                slice_embeds = self.visual_encoder(slice_image).to(self.llama_model_device)

                # if modality[0] == "brainMRI":
                #     slice_embeds = self.visual_encoder(slice_image).to(self.llama_model_device)
                #     #print('bmri slice_embeds.shape', slice_embeds.shape
                # elif modality[0] == "rectalMRIVQA":
                #     slice_embeds = self.visual_encoder_two(slice_image).to(self.llama_model_device)

                # elif modality[0] == "brainMRIVQA":
                #     slice_embeds = self.visual_encoder(slice_image).to(self.llama_model_device)
                #     #print('bmri vqa slice_embeds.shape', slice_embeds.shape)
                # elif modality[0] == "rectalMRI":
                #     slice_embeds = self.visual_encoder_two(slice_image).to(self.llama_model_device)
                #     #print('rmri slice_embeds.shape', slice_embeds.shape)

                slice_embeds = slice_embeds.unsqueeze(dim=1)
                image_embeds = torch.cat((image_embeds, slice_embeds), dim=1)
            #print('image_embeds.shape', image_embeds.shape)
            #print('attn_mask.shape', attn_mask.shape)
            print('image encoding END')
            image_embeds = image_embeds.to(self.llama_model_device)
            #z_attention_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[1]).to(self.llama_model_device)
            z_attention_mask = attn_mask
            z_output = self.Zformer(inputs_embeds=image_embeds, attention_mask=z_attention_mask)

            #print('z_output.last_hidden_state.shape', z_output.last_hidden_state.shape)
            cls_output = z_output.last_hidden_state[:,:,:]#.unsqueeze(1)
            #cls_global = cls_output.mean(dim=1).unsqueeze(1)
            image_embeds = self.qform_proj_chz(cls_output)
            image_atts = z_attention_mask.to(self.llama_model_device) #torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.llama_model_device)
            #print('self.query_tokens.shape', self.query_tokens.shape)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        ###FOR VISION ENCODER + FFdef encode_imgN/LINEAR LAYER Only

        return query_output.last_hidden_state

    def encode_img_m(self, image, attn_mask, modality):
        #device = self.llama_model_device
        #print('Device', device)

        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():

            image_embeds = torch.tensor([]).to(self.llama_model_device)
            slice_embeds = torch.tensor([]).to(self.llama_model_device)

            for idx in range(image.shape[1]):

                slice = image[:,idx,:,:,:]
                slice_image = kornia.geometry.transform.resize(slice, size=(224, 224))
                slice_embeds = self.visual_encoder(slice_image).to(self.llama_model_device)

                # if modality[0] == "brainMRI":
                #     slice_embeds = self.visual_encoder(slice_image).to(self.llama_model_device)
                #     #print('bmri slice_embeds.shape', slice_embeds.shape)

                # elif modality[0] == "rectalMRI":
                #     slice_embeds = self.visual_encoder_two(slice_image).to(self.llama_model_device)
                #     #print('rmri slice_embeds.shape', slice_embeds.shape)

                slice_embeds = slice_embeds.unsqueeze(dim=1)
                #print('slice_embeds.shape', slice_embeds.shape)
                image_embeds = torch.cat((image_embeds, slice_embeds), dim=1)
                #print('cat image_embeds.shape', image_embeds.shape)


            #print("image_embeds.shape", image_embeds.shape)
            image_embeds = image_embeds.to(self.llama_model_device)

            #print("image_embeds.shape", image_embeds.shape)
            z_attention_mask = attn_mask #torch.ones(image_embeds.shape[0], image_embeds.shape[1]).to(self.llama_model_device)
            z_output = self.Zformer(inputs_embeds=image_embeds, attention_mask=z_attention_mask)
            cls_output = z_output.last_hidden_state[:,:,:]

            image_embeds = self.qform_proj_chz(cls_output)

            image_atts = z_attention_mask.to(self.llama_model_device) #torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.llama_model_device)

            query_tokens_m = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output_m = self.Qformer_m.bert(
                query_embeds=query_tokens_m,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        return query_output_m.last_hidden_state

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
            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def prompt_vqa_wrap(self, img_embeds, atts_img, prompt):
        p_after_embeds = torch.tensor([]).to(img_embeds.device)
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


    def forward(self, samples):

        #loss_itc = torch.tensor(0.).to(self.llama_model_device)
        # loss
        # with torch.no_grad():
        #     self.temp.clamp_(0.001, 0.5)


        image = samples[0]#.to(self.llama_model_device)
        text_s = samples[1] #.to(self.llama_model_device)
        modality = samples[2]
        attn_mask = samples[4]
        bs, ds, c, h, w = image.size()
        #print('image', image.size())
        #bs, ds, c, h, w = image.size()
        #bs, c, h, w = image.size()
        #image = image.view(-1, c, h, w)
        #print('modality')
        query_output = self.encode_img(image, attn_mask, modality)
        #print('img_embeds.shape after Q-former', img_embeds.shape)
        #print('query_output.shape', query_output.shape)
        #atts_img = atts_img.to(self.llama_model_device)

        ###MODIFY THIS PART FOR VQA GENERATION###


        #img_embeds = img_embeds + self.interpolate_pos_encoding(self.z_embed, img_embeds)

        ###==========Image-Text Alignment===============###
        # ###Use bert-base-uncased tokenizer
        # #with torch.no_grad():
        # text = [t + self.end_sym for t in text_s]
        # #print('len(text)', len(text))
        # txt_tokens = self.tokenizer(text, return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len).to(self.llama_model_device)

        # print('txt_tokens.input_ids.shape', txt_tokens.input_ids.shape)
        # #txt_atts = torch.ones(txt_tokens.input_ids.size()[:-1], dtype=torch.long).to(self.llama_model_device)

        # txt_output = self.Qformer.bert(
        #             input_ids= txt_tokens.input_ids,
        #             attention_mask= txt_tokens.attention_mask,
        #             encoder_hidden_states=None,
        #             encoder_attention_mask=None,
        #             return_dict=True,
        #             is_decoder=False
        #         )

        # text_embeds = txt_output.last_hidden_state

        # # print('query_output.shape', query_output.shape)
        # print('text_embeds.shape', text_embeds.shape)

        # image_feats = F.normalize(query_output, dim=-1)
        # text_feat = F.normalize(text_embeds[:,0,:], dim=-1)
        # #print('image_feats.shape', image_feats.shape)
        # #print('text_feat.shape', text_feat.shape)
        # #print('image_feats.unsqueeze(1).shape', image_feats.unsqueeze(1).shape)
        # #print('text_feat.unsqueeze(-1).shape', text_feat.unsqueeze(-1).shape)
        # sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat.unsqueeze(-1)).squeeze()
        # print('sim_q2t', sim_q2t.shape)
        # sim_i2t, ind_i2t = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp
        # #print('sim_q2t, sim_i2t', sim_q2t.shape, sim_i2t.shape)
        # sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        # print('sim_t2q', sim_t2q.shape)
        # sim_t2i, ind_t2i = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp
        # #print('s   im_t2q, sim_t2i', sim_q2t.shape, sim_i2t.shape)
        # targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)
        # #print('targets', targets.shape)
        # print('sim_i2t', sim_i2t, targets)
        # print('sim_t2i', sim_t2i, targets)
        # # print('sim_t2q, sim_t2i', sim_i2t.shape, sim_t2i.shape)
        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #     + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        # ) / 2



        ###Image-Text Alignment with MoCo###

        text = [t + self.end_sym for t in samples[1]]


        ###Use bert-base-uncased tokenizer
        txt_tokens = self.tokenizer(text, return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len).to(self.llama_model_device)

        #txt_atts = torch.ones(txt_tokens.input_ids.size()[:-1], dtype=torch.long).to(self.llama_model_device)

        txt_output = self.Qformer.bert(
                    input_ids= txt_tokens.input_ids,
                    attention_mask= txt_tokens.attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    return_dict=True,
                    is_decoder=False
                )

        txt_embeds = txt_output.last_hidden_state[:,0,:]
        #print('txt_embeds.shape', txt_embeds.shape)
        #print('query_output.shape', query_output.shape)
        img_feat = F.normalize(query_output.mean(dim=1), dim=-1)
        txt_feat = F.normalize(txt_embeds, dim=-1)

        ###=======MoCo=========###

        with torch.no_grad():
            self._momentum_update()
            query_output_m = self.encode_img_m(image, attn_mask, modality)
            #img_embeds_m = img_embeds_m.view(bs, ds, num_patch, embed_dim)
            #img_embeds_m = img_embeds_m + self.interpolate_pos_encoding(self.z_embed_m, img_embeds_m)

            #img_embeds_m = img_embeds_m.view(bs, -1, embed_dim)
            #print('query_output_m.shape', query_output_m.shape)
            image_feat_m = F.normalize(query_output_m.mean(dim=1), dim=-1)
            #print('txt_tokens.attention_mask', txt_tokens.attention_mask)
            text_output_m = self.Qformer_m.bert(
                    input_ids= txt_tokens.input_ids,
                    attention_mask= txt_tokens.attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    return_dict=True,
                    is_decoder=False
                )
            #txt_embeds_m = text_output_m.last_hidden_state[:,0,:]
            text_feat_m = F.normalize(text_output_m.last_hidden_state[:, 0, :], dim=-1)

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # image-text alignment (diagonal)
            #print('image_feat_m', image_feat_m.shape)
            #print('image_feat_all', image_feat_all.shape)
            #print('text_feat_m', text_feat_m.shape)
            #print('text_feat_all', text_feat_all.shape)
            #print('image_feat_all', image_feat_all.shape)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            #print('sim_i2t_m.shape', sim_i2t_m.shape)
            #print('sim_t2i_m.shape', sim_t2i_m.shape)
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image[0].device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = self.alphas * F.softmax(sim_i2t_m, dim=1) + (1 - self.alphas) * sim_targets
            sim_t2i_targets = self.alphas * F.softmax(sim_t2i_m, dim=1) + (1 - self.alphas) * sim_targets

            # intramodal alignment (diagonal)
            sim_i2i_m = image_feat_m @ image_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp
            #print('sim_i2i_m.shape', sim_i2i_m.shape)
            #print('sim_t2t_m.shape', sim_t2t_m.shape)
            sim_i2i_targets = self.alphas * F.softmax(sim_i2i_m, dim=1) + (1 - self.alphas) * sim_targets
            sim_t2t_targets = self.alphas * F.softmax(sim_t2t_m, dim=1) + (1 - self.alphas) * sim_targets

        sim_i2t = img_feat @ text_feat_all / self.temp
        sim_t2i = txt_feat @ image_feat_all / self.temp

        _, ind_i2t = sim_i2t.max(-1)
        _, ind_t2i = sim_t2i.max(-1)
        print('sim_i2t.shape', sim_i2t.shape, 'sim_i2t_targets.shape', sim_i2t_targets.shape)
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        sim_i2i = img_feat @ image_feat_all / self.temp
        sim_t2t = txt_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        #print('query_output.shape', query_output.shape)
        # image_feats = F.normalize(query_output.mean(dim=1), dim=-1)
        # #print('text_feat.shape', text_embeds.shape)
        # #text_feat = F.normalize(text_embeds[:,0,:], dim=-1)
        # text_feat = F.normalize(text_embeds.mean(dim=1), dim=-1)
        # sim_q2t = torch.matmul(image_feats, text_feat.t())
        # sim_i2t = sim_q2t / self.temp
        # _, ind_i2t = sim_i2t.max(-1)
        # sim_t2q = torch.matmul(text_feat, image_feats.t())
        # sim_t2i = sim_t2q / self.temp
        # _, ind_t2i = sim_t2i.max(-1)

        # targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)
        # print('ind_i2t, targets', ind_i2t, targets)
        # print('ind_t2i, targets', ind_t2i, targets)
        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets, label_smoothing=0.5)
        #     + F.cross_entropy(sim_t2i, targets, label_smoothing=0.5)
        # ) / 2

        ### Masked Language Modeling ###
        print('ind_i2t', ind_i2t)
        print('ind_t2i', ind_t2i)
        print('loss_itc: ', loss_itc.item())
        return {"loss": loss_itc}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", 'vit_base') #"ViT-B/32" #"eva_clip_g"
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token", 8)
        llama_model = cfg.get("llama_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", True)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
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

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
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
            queue_size=queue_size
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
