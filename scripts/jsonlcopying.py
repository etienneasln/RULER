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

def copy_jsonl_data_for_task(source_file, target_directory,target_file):
    target_file_path=target_directory+"/"+target_file
    samples=read_manifest(source_file)
    os.makedirs(target_directory,exist_ok=True)
    with open(target_file_path, 'w', encoding='utf-8') as tgt:
        for sample in samples:

            input=sample['input']
            prompt=input

            map={"prompt":prompt}
            
            outputs=sample['outputs']
            
            match task:
                case 'niah_single_1'|'niah_single_2'|'niah_single_3'|'niah_multikey_1'|'niah_multikey_2'|'niah_multikey_3':
                    output=outputs[0]
                    concat=input+" "+output+"."
                    answer=output+'.'
                    map["answer_0"]=answer
                    map["concatenation_0"]=concat
                case 'vt'|'fwe':
                    answer=" ".join(outputs)
                    concat=input+" "+answer
                    map["answer_0"]=answer
                    map["concatenation_0"]=concat
                case 'qa_1'|'qa_2':
                    for i in range(len(outputs)):
                        output=outputs[i]
                        concat=input+" "+output+"."
                        answer=output+'.'
                        map[f"answer_{i}"]=answer
                        map[f"concatenation_{i}"]=concat
                case 'cwe':
                    answerlist=[]
                    for i in range(len(output)):
                        output=outputs[i]
                        answerlist.append(f"{i+1}. {output}")
                    answer=" ".join(answerlist)
                    concat=input+" "+answer
                    map["answer_0"]=answer
                    map["concatenation_0"]=concat
            json.dump(map,tgt)
            tgt.write('\n')


if __name__ == "__main__":
    single_copy=args.single_copy
    if single_copy:
        contextlengths=[args.context_length]
    else:
        contextlengths=[1024,2048,4096,8192,16384,32768]
    model_path=args.model_name_or_path
    task=args.task_name
    target_file= 'example_'+task+'.jsonl'
    for contextlength in contextlengths:    
        strcontextlength=str(contextlength)
        source_file_path = '../results/'+model_path+'/synthetic/'+strcontextlength+'/data/'+task+'/validation.jsonl'
        target_directory='pred/examples/'+model_path+'/synthetic/'+strcontextlength
        copy_jsonl_data_for_task(source_file_path, target_directory,target_file)
