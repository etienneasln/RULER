# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
import sys

from typing import Dict, List, Optional


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results

    # def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
    #     output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
    #     assert len(output) == 1
    #     generated_text = output[0]["generated_text"]
            
    #     # remove the input form the generated text
    #     if generated_text.startswith(prompt):
    #         generated_text = generated_text[len(prompt) :]
                
    #     if self.stop is not None:
    #         for s in self.stop:
    #             generated_text = generated_text.split(s)[0]
    #     return {'text': [generated_text]}


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]





class LandmarkAttentionModel:
    
    path_to_landmark_attention="../models/landmark-attention-repo/"
    path_to_fined_tuned_weights=path_to_landmark_attention+"llama/llama_weights/7B_hf"
    llama_weights_7b_tuned=path_to_fined_tuned_weights
    use_flash = False 
    top_k=5
    cache_path = path_to_landmark_attention+"llama/hf-cache/"
    dtype = torch.bfloat16
    
    def __init__(self, name_or_path, **generation_kwargs) -> None:
        # print(LandmarkAttentionModel.path_to_landmark_attention+"llama")
        
        sys.path.insert(1,LandmarkAttentionModel.path_to_landmark_attention+"llama")    
        from llama_mem import LlamaForCausalLM
        # from client_wrappers import LlamaForCausalLM
        import transformers

        
        # model_kwargs = {"attn_implementation": "flash_attention_2"}

        model = LlamaForCausalLM.from_pretrained(
            name_or_path,
            cache_dir=LandmarkAttentionModel.cache_path,
            torch_dtype=LandmarkAttentionModel.dtype,
        )

        # model.to('cuda:0')
        # model.to('cpu')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            name_or_path,
            cache_dir=LandmarkAttentionModel.cache_path,
            model_max_length=model.config.train_context_length,
            padding_side="right",
            use_fast=False,
        )


        mem_id = self.tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        from transformers import pipeline

        self.pipeline = pipeline("text-generation",
                                model=model,
                                tokenizer=self.tokenizer, 
                                device=model.device,
                                offload_cache_to_cpu=LandmarkAttentionModel.use_flash, 
                                use_flash=LandmarkAttentionModel.use_flash, 
                                cache_top_k=LandmarkAttentionModel.top_k,
                                # model_kwargs=model_kwargs,
                                )
        

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        return self.process_batch([prompt], **kwargs)[0]

        
    
    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        
        output = self.pipeline(text_inputs=prompts,num_return_sequences=1,max_new_tokens=10)
        assert len(output) == len(prompts)
        generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results

    # def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
    #     output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
    #     assert len(output) == 1
    #     generated_text = output[0]["generated_text"]
            
    #     # remove the input form the generated text
    #     if generated_text.startswith(prompt):
    #         generated_text = generated_text[len(prompt) :]
                
    #     if self.stop is not None:
    #         for s in self.stop:
    #             generated_text = generated_text.split(s)[0]
    #     return {'text': [generated_text]}
