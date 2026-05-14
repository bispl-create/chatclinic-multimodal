import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from vllama.common.registry import registry
from vllama.models.blip2 import Blip2Base, disabled_train
from vllama.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import vllama.models.vision_transformer as vits
import pdb
import kornia

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)


@registry.register_model("vllama")
class vllama(Blip2Base):
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
        low_resource=False,  # use 8 bit and put vit in cpu =
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vit_path = "",
        lora_r = 0,
        lora_target_modules=["q_proj","v_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05
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

        ###FOR CheXZero
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

        ###FOR DINO
        '''
        self.visual_encoder =self.init_DINO_encoder(vit_model, vit_path, patch_size=16)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            logging.info("freeze vision encoder")

        print('Loading VIT Done')


        '''
        ###USING Q-Former

        # print('Loading Q-Former')
        # print('visual_encoder.num_features', self.visual_encoder.num_features)

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, 1408
        # )

        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)
        # self.Qformer = self.Qformer.train()

        #
        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False ###Fine-tune QFormer with CheXzero vision encoder
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.query_tokens.requires_grad = False
        #     logging.info("freeze Qformer")

        # print('Loading Q-Former Done')

        ###END Q-Former

        print('Loading LLAMA')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        #self.llama_tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        #self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            print("Low_resource activated")

            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={'': device_8bit}
            )

            #self.llama_model = transformers.AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, load_in_8bit=True, )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16
            )

        self.llama_model_device = self.llama_model.device

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

            self.llama_model.print_trainable_parameters()

        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('LLaMA FROZEN')
        print(self.llama_model)
        print(self.llama_model.model)
        #self.llama_model.model.model.embed_tokens.weight.requires_grad = False
        self.llama_model.model.embed_tokens.weight.requires_grad = False
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

        ###TODO - FIX QFormer Alignment self.llama_proj
        '''
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        '''
        #print('1-0', self.llama_model_device)
        ###Dimension Matching? Distribution Matching? ###REFER to Cross-Modal Finetuning
        #print('config.hidden_size', self.llama_model.config.hidden_size)
        #.
        #qform_proj = self.visual_encoder.num_features
        #qform_out = self.Qformer.bert.encoder.layer.0.crossattention.self.value.weight.shape[1]
        self.qform_proj_chz = nn.Linear(768, 1408).to(self.llama_model_device)
        self.llama_proj_chz = nn.Linear(768, self.llama_model.config.hidden_size).to(self.llama_model_device)
        #print('get device', self.llama_proj_chz.device)
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

    def vit_to_cpu(self):
        #self.ln_vision.to("cpu")
        #self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = 'cuda:0'
        #print('Device', device)

        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")


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
            image_embeds = torch.unsqueeze(image_embeds, dim=0)
            #image = [B, C, H, W] B we want it to be the number of depth slices...
            #For sorted images, ct.h5 -> [ 24, 58, 42, ...]

            #print("image_embeds.shape", image_embeds.shape)

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

            #print("query_output.shape", query_output.shape)
            #print("query_output.last_hidden_state", query_output.last_hidden_state.shape)
            #print('query max, min', query_output.last_hidden_state.max(), query_output.last_hidden_state.min())
            inputs_llama = self.llama_proj_chz(query_output.last_hidden_state)
            #print("inputs_llama.shape", inputs_llama.shape)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        ###FOR VISION ENCODER + FFN/LINEAR LAYER Only
        '''
        #with self.maybe_autocast():
        print('image.shape', image.shape)
        image_embeds = self.visual_encoder(image).to(device)
        image_cls_tokens = (image_embeds-image_embeds.min())/(image_embeds.max()-image_embeds.min()) #image_cls_tokens = image_embeds[:,-1,:]
        print('image_cls_tokens.shape', image_cls_tokens.shape)
        inputs_llama = self.llama_proj_chz(image_cls_tokens).to(device)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        inputs_llama = inputs_llama.unsqueeze(dim=1)
        atts_llama = atts_llama.unsqueeze(dim=1)
        '''
        return inputs_llama, atts_llama

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
        #image = samples["image"]
        #image = samples["image"]
        ###FOR CT
        #print(samples[0].shape, samples[1])
        image = samples[0]
        text = samples[1]
        #bs, ds, c, h, w = image.size()
        bs, c, h, w = image.size()
        image = image.view(-1, c, h, w)

        img_embeds, atts_img = self.encode_img(image)
        atts_img = atts_img.to(image.device)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)


        self.llama_tokenizer.padding_side = "right"

        #text = [t + self.end_sym for t in samples["text_input"]]
        text = [t + self.end_sym for t in samples[1]]
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

            loss = outputs.loss

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
            lora_dropout=lora_dropout
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
