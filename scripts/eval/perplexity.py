import torch 
from model_wrappers import HuggingFaceModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


if __name__ == "__main__":
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
    example_path="example.jsonl"
    data=read_manifest(example_path)
    for sample in data:
        answer=sample["answer"]
        answertokens=llm.tokenizer(answer,return_tensors="pt",add_special_tokens=False)['input_ids']
        labels = answertokens.clone()
        labels[:, :-1] = answertokens[:, 1:]
        labels[:, -1] = -100  # We ignore the loss on the last token prediction

        # Forward pass to get the loss
        outputs = llm(answertokens, labels=labels)
        loss = outputs.loss
        print(loss)