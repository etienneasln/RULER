import json
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True, help="Name or path of the model that we copy the expected results for")
parser.add_argument("--task_name", type=str,required=True,help="Task we copy the data for")
parser.add_argument("--single_copy",type=bool,default=False,help="If activated, then data for only one context length will be copied. Otherwise the data for all possible context lengths will be copied")
parser.add_argument("--context_length",type=int,default=16384,help="Context length of the data to be copied if single_copy activated")

args=parser.parse_args()

def copy_jsonl_data(source_file, target_directory,target_file):
    """
    Copy all data from source JSONL file to target JSONL file.
    
    :param source_file: Path to the source JSONL file.
    :param target_file: Path to the target JSONL file.
    """
    target_file_path=target_directory+"/"+target_file
    samples=read_manifest(source_file)
    os.makedirs(target_directory,exist_ok=True)
    with open(target_file_path, 'w', encoding='utf-8') as tgt:
        for sample in samples:
            concat=sample['input']+" "+sample['outputs'][0]+'.'
            questionindex=concat.index('?')
            prompt=concat[:questionindex+1]
            answer=concat[questionindex+2:]
            map={"prompt":prompt,"answer":answer,"concatenation":concat}
            json.dump(map,tgt)
            tgt.write('\n')


if __name__ == "__main__":
    single_copy=args.single_copy
    if single_copy:
        contextlengths=[args.context_length]
    else:
        contextlengths=[4096,8192,16384]
    model_path=args.model_name_or_path
    task=args.task_name
    target_file= 'example_'+task+'.jsonl'
    for contextlength in contextlengths:    
        strcontextlength=str(contextlength)
        source_file_path = '../results/'+model_path+'/synthetic/'+strcontextlength+'/data/'+task+'/validation.jsonl'
        target_directory='pred/examples/'+model_path+'/synthetic/'+strcontextlength
        copy_jsonl_data(source_file_path, target_directory,target_file)
