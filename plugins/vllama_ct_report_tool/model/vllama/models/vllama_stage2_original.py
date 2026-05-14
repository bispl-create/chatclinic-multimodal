import logging
import random
import os
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity  # Import Profiler
from vllama.common.registry import registry
from vllama.models.blip2 import Blip2Base, disabled_train
from vllama.models.modeling_llama import LlamaForCausalLM
from vllama.models.perceiver import PerceiverResampler
from transformers import LlamaTokenizer
from torchvision import transforms as pth_transforms
import torch.distributed as dist

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def masked_mae_loss(original, reconstructed, masks):
    # Apply mask to the outputs to focus the loss calculation on masked elements
    with record_function("Masked_MAE_Loss"):  # Profiling the loss function
        loss = torch.abs(original - reconstructed) * masks
        return loss.sum() / masks.sum()

@torch.no_grad()
def concat_all_gather(tensor):
    with record_function("Concat_All_Gather"):  # Profiling the gather operation
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

def forward_hook(module, input, output):
    module.used = True

def hook_function(module, grad_in, grad_out):
    print(f"Grad in: {grad_in[0].norm()}, Grad out: {grad_out[0].norm()}")

@registry.register_model("vllama_stage2_test")
class vllamastage2test(Blip2Base):
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
        evaluate=False
    ):
        super().__init__()

        self.prompt_template = prompt_template
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.evaluate = evaluate
        self.heads_to_use = heads_to_use

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

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)  #AutoTokenizer.from_pretrained(llama_model)
        #self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
            # self.llama_model = AutoModelForCausalLM.from_pretrained(
            #     llama_model,
            #     torch_dtype=torch.bfloat16,
            #     device_map={'': llm_device}
            # )

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
                print('type of param: ', name, param.dtype)
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
        self.llama_proj_chz.register_backward_hook(hook_function)
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
        self.prompt_dict["CTRATE"] = []
        for elem in self.prompt_list:
            self.prompt_dict["CTRATE"].append(elem)

        #vqa_prompt = '<Img><ImageHere></Img><Mask><MaskHere></Mask>'
        self.vqa_nomask_prompt = '<Img><ImageHere></Img>'
        #self.vqa_prompt = '<Img><ImageHere></Img><Mask><MaskHere></Mask>'
        self.vqa_nomask_template = prompt_template #.format(vqa_nomask_prompt)
        #self.vqa_template = prompt_template.format(vqa_prompt)


    def mask_embeddings(self, image_embeds, mask_prob=0.3):
        with record_function("Mask_Embeddings"):  # Profiling masked embedding generation
            batch_size, seq_length, embed_dim = image_embeds.shape
            mask = torch.rand(batch_size, seq_length, 1) < mask_prob
            mask = mask.to(self.llama_model_device)
            original_embeds = image_embeds.clone()
            image_embeds = image_embeds.masked_fill(mask, 0)
            return image_embeds, original_embeds, mask


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
            p_len = p_before_embeds.shape[1] + img_embeds.shape[1]
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            # wrapped_atts_img_before = atts_img[:, :1].expand(-1, p_len)
            # wrapped_atts_img_after = torch.zeros((batch_size, p_after_embeds.shape[1]), device=wrapped_atts_img_before.device)
            # wrapped_atts_img = torch.cat([wrapped_atts_img_before, wrapped_atts_img_after], dim=1)
            # print('wrapped_atts_img:', wrapped_atts_img)
            # print('wrapped_atts_img_shape:', wrapped_atts_img.shape)
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

    def encode_img(self, image, attn_mask):
        with record_function("Encode_Image"):  # Profiling the image encoding process
            slice_embeds_list = []
            for idx in range(image.shape[1]):
                slice_image = self.transform(image[:, idx, :, :, :])
                slice_embeds = self.visual_encoder(slice_image).unsqueeze(dim=1)
                slice_embeds_list.append(slice_embeds)
            image_embeds = torch.cat(slice_embeds_list, dim=1)

            z_output = self.Zformer(inputs_embeds=image_embeds, attention_mask=attn_mask)
            recon_embeds = self.perc_proj_chz(z_output.last_hidden_state[:, :, :])
            query_embeds = self.perceiver(recon_embeds).squeeze(1)
            inputs_llama = self.llama_proj_chz(query_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.llama_model_device)

        return inputs_llama, atts_llama

    def forward(self, samples):
        image = samples[0].to(self.llama_model_device)
        attn_mask = samples[3].to(self.llama_model_device)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:  # Profiling forward pass
            with record_function("Model_Forward"):  # Marking the forward pass for profiling
                img_embeds, atts_img = self.encode_img(image, attn_mask)

                text_s = samples[1]
                modality = samples[2]
                if modality[0] == 'CTRATE':
                    prompt = random.choice(self.prompt_dict['CTRATE'])
                    img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                    text = [t + self.end_sym for t in text_s]

                elif modality[0][:12] == 'CTRATEGround':

                    ###SELECTING APPROPRIATE PROMPTS###
                    #print('VQA Batch')
                    question = text_s[0]
                    answer = text_s[1]
                    print('q/a', question, answer)
                    question_prompt = [self.vqa_nomask_template.format('<Img><ImageHere></Img>'+" "+q) for q in question]
                    img_embeds, atts_img = self.prompt_vqa_nomask_wrap(img_embeds, atts_img, question_prompt)
                    text = [t + self.end_sym for t in answer]
                    print('prompt/qa', question_prompt, text)

                else:
                    text = ""

                self.llama_tokenizer.padding_side = "right"

                to_regress_tokens = self.llama_tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(self.llama_model_device)

                targets_pre = to_regress_tokens.input_ids.masked_fill(
                    to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                targets = targets_pre.repeat(atts_img.shape[0] // targets_pre.shape[0], 1)

                bos = torch.ones([img_embeds.shape[0], 1], dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.llama_model.model.model.embed_tokens(bos)
                atts_bos = atts_img[:, :1]

                to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
                inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
                attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

                llm_loss = outputs.loss

            print(prof.key_averages().table(sort_by="cpu_time_total"))  # Print profiling results

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
            evaluate=evaluate
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
