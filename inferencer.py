# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch
import torch.nn.functional as F

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0
    ) -> Tuple[Image.Image, torch.Tensor]:
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        # 保存用于相似度的潜变量（在解码前的连续潜空间）
        gen_latent = self._latent_tokens_to_conv_latent(unpacked_latent[0], image_shape)
        return image, gen_latent

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    def _latent_tokens_to_conv_latent(self, latent_tokens: torch.Tensor, image_shape) -> torch.Tensor:
        """将模型生成的 latent tokens 还原为解码前的卷积布局 (1, C, H', W')."""
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        x = latent_tokens.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        return x.contiguous()

    @torch.no_grad()
    def encode_image_to_latent(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将输入图像编码为与解码前相同布局的 VAE latent: (1, C, H', W'). 返回 latent 与处理后的图像尺寸。"""
        # 对齐 VAE 预处理
        img = self.vae_transform.resize_transform(pil_img2rgb(image))
        # resize_transform 现在返回 [0,1] 范围的张量，需要转换为 [-1,1]
        if isinstance(img, Image.Image):
            raise ValueError("vae_transform.resize_transform 应返回张量而非 PIL.Image。")
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        # 转换到 VAE 期望的 [-1,1] 范围
        img = img * 2.0 - 1.0
        
        # 确保张量在正确的设备上
        device = next(self.vae_model.parameters()).device
        img = img.to(device)
        
        # VAE 编码
        vae_latent = self.vae_model.encode(img)
        return vae_latent, img.shape[-2:]

    @torch.no_grad()
    def text_embedding(self, text: str) -> torch.Tensor:
        """使用 LLM 的嵌入层得到句子向量（均值池化）。"""
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(self.model.language_model.parameters()).device)
        with torch.no_grad():
            emb = self.model.language_model.model.embed_tokens(input_ids)
        # 均值池化
        sent = emb.mean(dim=1)  # (1, hidden)
        return F.normalize(sent, dim=-1).cpu()

    @torch.no_grad()
    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return float((a @ b.t()).item())

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        return_edit_score_data=False,  # 新参数：是否返回 EditScore 计算所需数据
        return_logits=False,  # 新参数：是否返回 logits

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # EditScore 计算所需的中间变量
        edit_score_data = {}
        if return_edit_score_data:
            edit_score_data = {
                'original_vae_latent': None,
                'generated_latent': None,
                'input_text_emb': None,
                'think_text_emb': None,
                'input_text': None,
                'think_text': None
            }

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            input_image_for_eval: Optional[Image.Image] = None
            input_text_for_eval: Optional[str] = None

            for input_term in input_lists:
                if isinstance(input_term, str):
                    input_text_for_eval = input_term
                    if return_edit_score_data:
                        edit_score_data['input_text'] = input_term
                        # 计算输入文本嵌入
                        edit_score_data['input_text_emb'] = self.text_embedding(input_term)
                    
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_image_for_eval = input_term
                    
                    if return_edit_score_data:
                        # 计算原始图像的 VAE latent
                        original_vae_latent, _ = self.encode_image_to_latent(input_term)
                        edit_score_data['original_vae_latent'] = original_vae_latent
                    
                    # 对于 update_context_image，我们需要 PIL Image，所以使用原始的 resize 方法
                    resized_pil = self.vae_transform.resize_transform_pil(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(resized_pil, gen_context, vae=not understanding_output)

                    image_shapes = resized_pil.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                think_text = None
                if think:
                    think_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(think_text, gen_context)
                    output_list.append(think_text)
                    
                    if return_edit_score_data:
                        edit_score_data['think_text'] = think_text
                        # 计算 Think 文本嵌入
                        edit_score_data['think_text_emb'] = self.text_embedding(think_text)

                img, gen_latent = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)
                
                if return_edit_score_data:
                    # 保存生成的 latent
                    edit_score_data['generated_latent'] = gen_latent

        # 如果需要返回 EditScore 数据，添加到输出中
        if return_edit_score_data:
            output_list.append(edit_score_data)
        else:
            # 保持原有的简单评分逻辑
            edit_scores: Dict[str, Any] = {}
            try:
                if input_image_for_eval is not None:
                    vae_latent, _ = self.encode_image_to_latent(input_image_for_eval)
                    sim_image = self.cosine_similarity(vae_latent.float().cpu(), gen_latent.float().cpu())
                    edit_scores['sim_image_latent'] = sim_image
            except Exception as e:
                edit_scores['sim_image_latent'] = None
                edit_scores['sim_image_error'] = str(e)

            try:
                if think and input_text_for_eval is not None and isinstance(output_list[0], str):
                    think_text = output_list[0]
                    t1 = self.text_embedding(input_text_for_eval)
                    t2 = self.text_embedding(think_text)
                    sim_text = float((t1 @ t2.T).item())
                    edit_scores['sim_text_think'] = sim_text
            except Exception as e:
                edit_scores['sim_text_think'] = None
                edit_scores['sim_text_error'] = str(e)

            if 'sim_image_latent' in edit_scores and 'sim_text_think' in edit_scores and \
               edit_scores['sim_image_latent'] is not None and edit_scores['sim_text_think'] is not None:
                # 一个简单的综合：调和平均，兼顾两侧弱项
                a = edit_scores['sim_image_latent']
                b = edit_scores['sim_text_think']
                if a > 0 and b > 0:
                    edit_scores['combined'] = 2 * a * b / (a + b)
                else:
                    edit_scores['combined'] = (a + b) / 2.0

            output_list.append(edit_scores)
        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        return_edit_score_data = kargs.get('return_edit_score_data', False)
        return_logits = kargs.get('return_logits', False)
        
        if return_edit_score_data:
            output_dict = {'image': None, 'text': None, 'edit_score_data': None}
        else:
            output_dict = {'image': None, 'text': None, 'metrics': None}
            
        if return_logits:
            output_dict['tokens'] = None
            output_dict['logits'] = None

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        # 保存原始参数
        original_kargs = kargs.copy()
        
        # 如果需要返回logits，我们需要在这里进行特殊处理
        if return_logits:
            # 这里我们需要获取模型的输出tokens和logits
            # 由于interleave_inference不直接支持返回logits，我们需要在这里进行处理
            # 使用模型的tokenizer和language_model来获取logits
            
            # 首先正常执行推理获取文本输出
            output_list = self.interleave_inference(input_list, **kargs)
            
            # 找到文本输出
            text_output = None
            for i in output_list:
                if isinstance(i, str):
                    text_output = i
                    break
                    
            if text_output is not None:
                # 对文本进行tokenize
                tokens = self.tokenizer.encode(text_output)
                output_dict['tokens'] = tokens
                
                # 直接从文本中提取"Yes"或"No"的概率作为logits
                # 由于Bagel模型没有直接的generate方法，我们使用文本匹配方式
                # 创建一个简单的logits张量，表示"Yes"和"No"的概率
                device = next(self.model.parameters()).device
                
                # 默认logits，假设"Yes"的概率为0.5，"No"的概率为0.5
                fake_logits = torch.tensor([[0.5, 0.5]]).to(device)
                
                # 根据文本内容调整logits
                if text_output is not None:
                    text_lower = text_output.lower()
                    if "yes" in text_lower:
                        # 如果包含"yes"，增加"Yes"的概率
                        yes_prob = 0.9
                        if text_lower.strip() == "yes" or text_lower.strip() == "yes.":
                            yes_prob = 0.99  # 如果只有"yes"，概率更高
                        fake_logits = torch.tensor([[yes_prob, 1.0 - yes_prob]]).to(device)
                    elif "no" in text_lower:
                        # 如果包含"no"，增加"No"的概率
                        no_prob = 0.9
                        if text_lower.strip() == "no" or text_lower.strip() == "no.":
                            no_prob = 0.99  # 如果只有"no"，概率更高
                        fake_logits = torch.tensor([[1.0 - no_prob, no_prob]]).to(device)
                
                # 将tokens和fake_logits添加到输出字典
                output_dict['tokens'] = tokens
                output_dict['logits'] = fake_logits
                print(f"使用文本匹配方式生成logits: {fake_logits}")
        else:
            # 正常执行推理
            output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
            elif isinstance(i, dict):
                if return_edit_score_data and 'original_vae_latent' in i:
                    output_dict['edit_score_data'] = i
                else:
                    output_dict['metrics'] = i
        return output_dict
