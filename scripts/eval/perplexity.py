import torch 

import sys
sys.path.insert(1,"../pred")

from model_wrappers import HuggingFaceModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

def main():
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"
    temperature=0.0
    top_k=1
    top_p=1.0
    stop_words=""
    tokens_to_generate=128
    #We load the LLM
    llm=HuggingFaceModel(
        name_or_path=model_name_or_path,
        do_sample=temperature > 0,
        repetition_penalty=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop=stop_words,
        max_new_tokens=tokens_to_generate,
    )
    example_path="../pred/example.jsonl"
    data=read_manifest(example_path)
    losses=[]
    model=llm.pipeline.model if llm.pipeline else llm.model
    tokenizer=llm.pipeline.tokenizer if llm.pipeline else llm.tokenizer
    for sample in tqdm(data):

        prompt=sample["prompt"]
        answer=sample["answer"]
        
        
        
        prompttokens=tokenizer(prompt,return_tensors="pt").to(model.device).input_ids
        answertokens=tokenizer(answer,return_tensors="pt",add_special_tokens=False).to(model.device).input_ids
        concattokens=torch.cat([prompttokens,answertokens],1)
        labels = concattokens.clone()
        
        lengthofprompt=len(prompttokens[0].tolist())
        labels[:,:lengthofprompt]=-100
        

        with torch.no_grad():
            outputs=model(concattokens, labels=labels)
        

        loss =outputs.loss
        losses.append(loss)
        print(loss)
    ppl=torch.exp(torch.stack(losses).mean())
    print(f"Perplexity:{ppl}")


if __name__ == "__main__":
    main()
