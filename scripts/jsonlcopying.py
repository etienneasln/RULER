import json
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
import os

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
    contextlengths=[4096,8192,16384]
    model_path='LargeWorldModel/LWM-Text-Chat-1M'
    task='niah_single_1'
    for contextlength in contextlengths:    
        strcontextlength=str(contextlength)
        source_file_path = '../results/'+model_path+'/synthetic/'+strcontextlength+'/data/'+task+'/validation.jsonl'
        target_file= 'example_'+task+'.jsonl'
        target_directory='pred/examples/'+model_path+'/synthetic/'+strcontextlength
        copy_jsonl_data(source_file_path, target_directory,target_file)
