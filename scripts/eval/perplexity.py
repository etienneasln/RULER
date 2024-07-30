import torch 

import sys
sys.path.insert(1,"../pred")
import argparse

from model_wrappers import HuggingFaceModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True, help="Name or path of the model that is to be evaluated")
parser.add_argument("--context_length", type=int, required=True,help="Context length of the prompts to evaluate perplexity on")
parser.add_argument("--task_name", type=str,required=True,help="Task we evaluate perplexity on")
parser.add_argument("--only_answer",type=bool,default=False,help="If activated, we evaluate perplexity without the prompt as context. Otherwise we evaluate with the whole prompt and answer as context")

args=parser.parse_args()

def main():
    
    model_name_or_path=args.model_name_or_path
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
    task=args.task_name
    contextlength=args.context_length
    strcontextlength=str(contextlength)
    example_path="../pred/examples/"+model_name_or_path+"/synthetic/"+strcontextlength+"/example_"+task+".jsonl"
    data=read_manifest(example_path)
    losses=[]
    model=llm.pipeline.model if llm.pipeline else llm.model
    tokenizer=llm.pipeline.tokenizer if llm.pipeline else llm.tokenizer
    device=model.device
    onlyanswer=args.only_answer
    # for nkeeplast in range(1,100):
    for sample in tqdm(data):

        prompt=sample["prompt"]
        answer=sample["answer"]
        
        
        
        prompttokens=tokenizer(prompt,return_tensors="pt").to(device).input_ids
        answertokens=tokenizer(answer,return_tensors="pt",add_special_tokens=False).to(device).input_ids
        concattokens=torch.cat([prompttokens,answertokens],1)

        if not onlyanswer:
            tokens = concattokens
            labels = tokens.clone()
            
            
            
            lengthofanswer=len(answertokens[0].tolist())
            labels[:,:-lengthofanswer]=-100

        #----------------------------------------
        # Evaluating perplexity when context does not include prompt 
        else:
            labels = answertokens.clone()
            tokens = answertokens
        #---------------------------------
        

        with torch.no_grad():
            outputs=model(tokens, labels=labels)
        
        loss = outputs.loss
        losses.append(loss)
        



    ppl=torch.exp(torch.stack(losses).mean())
    if onlyanswer:
        text="Average perplexity over the answer by passing only the answer as context"
    else:
        text="Average perplexity over the answer by passing all the tokens of the concatenation of the prompt and the answer as context"
    print(f"{text}:{ppl}")
    losses.clear()


if __name__ == "__main__":
    main()
