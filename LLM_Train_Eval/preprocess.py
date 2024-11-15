import sentencepiece as sp
import torch
import json
import os
from tqdm import tqdm
import pickle

class SkyPilePreprocess:

    def __init__(self,dst_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load("tokenizer.model")
        self._dst_dir = dst_dir

    def __call__(self,file_path):
        _base_name = os.path.basename(file_path)
        _fn = _base_name.split(".")[0]

        _vocs = []
        for _line in tqdm(open(file_path,"r+",encoding="UTF-8")):
            _txt_js = json.loads(_line)
            _ids = self._spm.Encode(_txt_js["text"])
            _vocs.extend(_ids)
        _vocs = torch.tensor(_vocs,dtype=torch.int16)
        print(_vocs.shape)

        torch.save(_vocs,f"{self._dst_dir}/{_fn}")

class RuozhibaPilePreprocess:

    def __init__(self,dst_dir):
        self._spm=sp.SentencePieceProcessor()
        self._spm.Load("tokenizer.model")
        self._dst_dir=dst_dir
        

    def __call__(self,file_path):
        _base_name=os.path.basename(file_path)
        _fn=_base_name.split(".")[0]

        with open(file_path,"r+",encoding="UTF-8") as fr:

            ''''
            load是从文件对象中读取json数据
            loads用于从包含json数据的字符串作为参数
            '''
            _datas=[]
            _js=json.load(fr)
            

            for _obj in _js:
                _user=_obj["instruction"]
                _assistant=_obj["output"]

                # _input=f"<|im_start|>system\n以下的问题用中文回答。<|im_end|><|im_start|>user\n{_user}<|im_end|><|im_start|>assistant\n"
                # _output=f"{_assistant}<|im_end|>"
                
                _bos_id=[self._spm.bos_id()]   
                _eos_id=[self._spm.eos_id()]
                _input_ids=(_bos_id+self._spm.Encode("system\n以下的问题用中文回答。")+_eos_id+
                            _bos_id+self._spm.Encode(f"user\n{_user}")+_eos_id+
                            _bos_id+self._spm.Encode("assistant\n"))
                _output_ids=(self._spm.Encode(f"{_assistant}")+_eos_id)

                _prompt=_input_ids+_output_ids
                _tag=len(_input_ids)*[0,]+_output_ids

                _datas.append([_prompt,_tag])
            with open(f"{self._dst_dir}/{_fn}.bin","wb") as fw:
                pickle.dump(_datas,fw)




if __name__ == '__main__':
    # preprocess = SkyPilePreprocess("datas")
    # preprocess("2022-21_zh_middle_0011.jsonl")
    sft=RuozhibaPilePreprocess('sft_datas')
    sft("./ruozhiba_qa.json")