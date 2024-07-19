import json
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

def copy_jsonl_data(source_file, target_file):
    """
    Copy all data from source JSONL file to target JSONL file.
    
    :param source_file: Path to the source JSONL file.
    :param target_file: Path to the target JSONL file.
    """
    samples=read_manifest(source_file)
    with open(target_file, 'w', encoding='utf-8') as tgt:
        for sample in samples:
            concat=sample['input']+" "+sample['outputs'][0]+'.'
            questionindex=concat.index('?')
            prompt=concat[:questionindex+1]
            answer=concat[questionindex+2:]
            map={"prompt":prompt,"answer":answer,"concatenation":concat}
            json.dump(map,tgt)
            tgt.write('\n')


if __name__ == "__main__":
    source_file_path = '../results/mistralai/Mistral-7B-Instruct-v0.2/synthetic/4096/pred/niah_single_1.jsonl'
    target_file_path = 'pred/example.jsonl'
    
    copy_jsonl_data(source_file_path, target_file_path)
