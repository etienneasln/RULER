import torch 

import sys
sys.path.insert(1,"../pred")

from model_wrappers import HuggingFaceModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

def main():
    
    model_name_or_path="LargeWorldModel/LWM-Text-Chat-1M"
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
    task='niah_single_1'
    contextlength=4096
    strcontextlength=str(contextlength)
    example_path="../pred/examples/"+model_name_or_path+"/synthetic/"+strcontextlength+"/example_"+task+".jsonl"
    data=read_manifest(example_path)
    losses=[]
    model=llm.pipeline.model if llm.pipeline else llm.model
    tokenizer=llm.pipeline.tokenizer if llm.pipeline else llm.tokenizer
    device=model.device
    # for nkeeplast in range(1,100):
    for sample in tqdm(data):

        prompt=sample["prompt"]
        answer=sample["answer"]
        
        
        
        prompttokens=tokenizer(prompt,return_tensors="pt").to(device).input_ids
        answertokens=tokenizer(answer,return_tensors="pt",add_special_tokens=False).to(device).input_ids
        concattokens=torch.cat([prompttokens,answertokens],1)

        tokens = concattokens
        labels = tokens.clone()
        
        
        
        lengthofanswer=len(answertokens[0].tolist())
        labels[:,:-lengthofanswer]=-100

        #----------------------------------------
        # For Mistral, 4096, niah_single_1
        # Evaluating perplexity when context does not include prompt --> average perplexity over the 500 samples is approximately 175
        # When context includes prompt, average perplexity is approximately 1.42
        # labels = answertokens.clone()
        # tokens = answertokens
        #---------------------------------
        

        with torch.no_grad():
            outputs=model(tokens, labels=labels)
        
        loss = outputs.loss
        losses.append(loss)
        



    ppl=torch.exp(torch.stack(losses).mean())
    # print(f"Perplexity for nkeeplast:{nkeeplast}:{ppl}")
    print(f"Perplexity:{ppl}")
    losses.clear()


if __name__ == "__main__":
    main()
