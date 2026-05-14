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



@registry.register_model("vllamaita_frozen")
class vllamaItaFrozen(Blip2Base):
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
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=64,
        end_sym='\n',
        low_resource=True,  # use 8 bit and put vit in cpu =
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vit_path = "/scratch/slurm-user3/changsun/dino/checkpoint/rectal_MRI_all_sorted_imgnet_pretrained/checkpoint0095.pth",
        lora_r = 0,
        lora_target_modules=["q_proj","v_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        temp=0,
        alpha=0,
        momentum=0,
        queue_size=500,
        embed_dim=768
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
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
        '''
        print("CHECK UPDATE")
        self.visual_encoder = self.init_CheXzero_encoder(vit_path)

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
        self.visual_encoder =self.init_DINO_encoder(vit_model, vit_path, patch_size=16)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            logging.info("freeze vision encoder")

        print('Loading VIT Done')

        ###USING Q-Former
        print('Loading Q-Former')
        print('visual_encoder.num_features', self.visual_encoder.num_features)

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)

        self.Qformer_m, _ = self.init_Qformer(num_query_token, 1408)
        #print(self.Qformer.parameters())
        self.Qformer.cls = None
        self.Qformer_m.cls = None
        #self.Qformer.bert.embeddings.word_embeddings = None
        #self.Qformer.bert.embeddings.position_embeddings = None
        '''
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        for layer in self.Qformer_m.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        '''
        self.load_from_pretrained(url_or_filename=q_former_model)
        self.Qformer = self.Qformer.train()


        if freeze_qformer:
            for layer in self.Qformer.bert.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False
            for name, param in self.Qformer_m.named_parameters():
                param.requires_grad = False ###Fine-tune QFormer with CheXzero vision encoder
            self.Qformer_m = self.Qformer_m.eval()
            self.Qformer_m.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        print('Loading Q-Former Done')

        ###END Q-Former

        print('Loading LLAMA')


        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # if self.low_resource:
        #     print("Low_resource activated")

        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         llama_model,
        #         load_in_8bit=True,
        #         torch_dtype=torch.float16,
        #         device_map={'': device_8bit}
        #     )

        # else:
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         llama_model,
        #         torch_dtype=torch.bfloat16
        #     )

        # self.llama_model_device = self.llama_model.device

        # if lora_r > 0:
        #     self.llama_model = prepare_model_for_int8_training(self.llama_model)
        #     loraconfig = LoraConfig(
        #         r=lora_r,
        #         bias="none",
        #         task_type="CAUSAL_LM",
        #         target_modules=lora_target_modules,
        #         lora_alpha=lora_alpha,
        #         lora_dropout=lora_dropout,
        #     )
        #     self.llama_model = get_peft_model(self.llama_model, loraconfig)

        #     self.llama_model.print_trainable_parameters()

        # else:
        #     for name, param in self.llama_model.named_parameters():
        #         param.requires_grad = False
        #     print('LLaMA FROZEN')

        # self.llama_model.model.embed_tokens.weight.requires_grad = False

        print('Loading LLAMA Done')

        '''Error: Int8 cannot be trained
        for name, param in self.llama_model.named_parameters():
            param.requires  _grad = True
            param.register_hook(lambda grad, name=name: print(f"Gradient for {name}: {grad}"))
        '''
        ''' Error:
            #param = param.type(torch.float64)
            #print('name:', name, 'param type', param.type)
            param.requires_grad = False
        '''

        #self.z_embed = nn.Parameter(torch.zeros(1, self.args.z_length, 1, vision_width))
        #self.z_embed_m = nn.Parameter(torch.zeros(1, self.args.z_length, 1, vision_width))
        ###TODO - FIX QFormer Alignment self.llama_proj
        '''
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        '''

        self.qform_proj_chz = nn.Linear(self.visual_encoder.num_features, 1408).to(self.llama_model_device)
        #self.llama_proj_chz = nn.Linear(768, self.llama_model.config.hidden_size).to(self.llama_model_device)

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
                            #[self.vision_proj, self.vision_proj_m],
                            #[self.qform_proj_chz, self.qform_proj_chz_m],
                            #[self.llama_proj_chz, self.llama_proj_chz_m],
                           # [self.text_proj, self.text_proj_m],
                            #[self.z_embed, self.z_embed_m],
                            # [self.zt_embed, self.zt_embed_m],
                            # [self.z_transformer, self.z_transformer_m],
                            # [self.overall_token, self.overall_token_m]
                            ]

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        #self.ln_vision.to("cpu")
        #self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
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


            '''
            try:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            except:
                model_pair[1].data = model_pair[1].data * self.momentum + model_pair[0].data * (1. - self.momentum)
            '''
    def encode_img(self, image, query_true=True):
        #device = self.llama_model_device
        #print('Device', device)

        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        device = 'cuda:0'
        with self.maybe_autocast():
            #image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            #image_embeds = self.visual_encoder(image).to(device)
            ###CTSEG-3D-PROCESSING
            #print('image.shape', image.shape)
            #image_embeds = []
            #print(image )
            for idx in range(image.shape[1]):
                slice = image[0][idx]
                #print('slice.shape', slice.shape)
                slice = kornia.geometry.transform.resize(slice, size=(224, 224))
                slice = torch.unsqueeze(slice, dim=0)
                slice_image = torch.cat((slice, slice, slice), dim=0)
                slice_image = torch.unsqueeze(slice_image, dim=0)
                slice_embeds = self.visual_encoder(slice_image).to(device)
                #print('slice embeds.shape', slice_embeds.shape)
                if idx == 0:
                    image_embeds = slice_embeds
                image_embeds = torch.cat((image_embeds, slice_embeds), dim=0)

            #print('image_embeds.shape', image_embeds.shape)
            image_embeds = image_embeds.to(device)
            image_embeds = torch.unsqueeze(image_embeds, dim=0)
            #image = [B, C, H, W] B we want it to be the number of depth slices...
            #For sorted images, ct.h5 -> [ 24, 58, 42, ...]

            #print("image_embeds.shape", image_embeds.shape)
            #print('image_embeds device', image_embeds.device)
            image_embeds = self.qform_proj_chz(image_embeds)
            #print('qform_aligned image_embeds.shape', image_embeds.shape)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            #print("image_atts.shape", image_atts.shape)
            #print("self.query_tokens.shape", self.query_tokens.shape)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            #print("query_tokens.shape", query_tokens.shape)
            query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            ###Freeze for Q-former Pre-alignment###
            #inputs_llama = self.llama_proj_chz(query_output.last_hidden_state)
            #atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

            query_output_m = self.Qformer_m.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

        return query_output.last_hidden_state, query_output_m.last_hidden_state ### inputs_llama, atts_llama,

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            #p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img


    def forward(self, samples):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        #image = samples["image"]
        #image = samples["image"]
        ###FOR CT
        #print(samples[0].shape, samples[1])
        image = samples[0]
        text = samples[1]
        bs, ds, c, h, w = image.size()
        #bs, c, h, w = image.size()
        image = image.view(-1, c, h, w)

        img_embeds, atts_img, query_output, query_output_m = self.encode_img(image, query_true=True)
        atts_img = atts_img.to(image.device)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        #print('img_embeds.shape', img_embeds.shape)
        #img_embeds = img_embeds + self.interpolate_pos_encoding(self.z_embed, img_embeds)

        ###==========Image-Text Alignment===============###
        #text = [t + self.end_sym for t in samples["text_input"]]
        text = [t + self.end_sym for t in samples[1]]

        ###Use bert-base-uncased tokenizer
        txt_tokens = self.tokenizer(text, return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len).to(image.device)

        #txt_atts = torch.ones(txt_tokens.input_ids.size()[:-1], dtype=torch.long).to(image.device)
        txt_output = self.Qformer.bert(
                    input_ids= txt_tokens.input_ids,
                    attention_mask= txt_tokens.attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    return_dict=True,
                    is_decoder=False
                )

        txt_embeds = txt_output.last_hidden_state[:,0,:]
        print('txt_embeds.shape', txt_embeds.shape)
        print('query_output.shape', query_output.shape)
        img_feat = F.normalize(query_output.mean(dim=1), dim=-1)
        txt_feat = F.normalize(txt_embeds, dim=-1)

        ###=======MoCo=========###
        with torch.no_grad():
            self._momentum_update()
            #_, _, query_output_m = self.encode_img(image, query_true=False)
            #img_embeds_m = img_embeds_m.view(bs, ds, num_patch, embed_dim)
            #img_embeds_m = img_embeds_m + self.interpolate_pos_encoding(self.z_embed_m, img_embeds_m)

            #img_embeds_m = img_embeds_m.view(bs, -1, embed_dim)
            print('query_output_m.shape', query_output_m.shape)

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
            txt_embeds_m = text_output_m.last_hidden_state[:,0,:]
            text_feat_m = F.normalize(text_output_m.last_hidden_state[:, 0, :], dim=-1)

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # image-text alignment (diagonal)
            print('image_feat_m, image_feat_all', image_feat_m.shape)
            print('image_feat_all', image_feat_all)
            print('image_feat_all', image_feat_all.shape)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            print('sim_i2t_m.shape', sim_i2t_m.shape)
            print('sim_t2i_m.shape', sim_t2i_m.shape)
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image[0].device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = self.alphas * F.softmax(sim_i2t_m, dim=1) + (1 - self.alphas) * sim_targets
            sim_t2i_targets = self.alphas * F.softmax(sim_t2i_m, dim=1) + (1 - self.alphas) * sim_targets

            # intramodal alignment (diagonal)
            sim_i2i_m = image_feat_m @ image_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp
            print('sim_i2i_m.shape', sim_i2i_m.shape)
            print('sim_t2t_m.shape', sim_t2t_m.shape)
            sim_i2i_targets = self.alphas * F.softmax(sim_i2i_m, dim=1) + (1 - self.alphas) * sim_targets
            sim_t2t_targets = self.alphas * F.softmax(sim_t2t_m, dim=1) + (1 - self.alphas) * sim_targets

        sim_i2t = img_feat @ text_feat_all / self.temp
        sim_t2i = txt_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        sim_i2i = img_feat @ image_feat_all / self.temp
        sim_t2t = txt_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)
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
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        #bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        #to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        to_regress_embeds = to_regress_embeds.repeat(img_embeds.shape[0]//to_regress_embeds.shape[0], 1, 1)

        #img_embeds = img_embeds / 10
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)

        to_regress_tokens.attention_mask = to_regress_tokens.attention_mask.repeat(atts_img.shape[0]//to_regress_tokens.attention_mask.shape[0], 1)

        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with torch.autograd.detect_anomaly():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

            lm_loss = outputs.loss
            q_loss = loss_ita
            loss = lm_loss + q_loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", 'vit_small') #"ViT-B/32" #"eva_clip_g"
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", True)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
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
        queue_size = cfg.get("queue_size", 10)

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
            momentum=momentum
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)


        return model
