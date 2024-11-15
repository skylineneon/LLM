from torch.utils.data import Dataset
import torch
import pickle


class PretrainDataset(Dataset):

    def __init__(self, file_path, seq):
        _ids = torch.load(file_path)
        print(_ids)
        print(_ids.dtype)
        _ids = _ids[:_ids.shape[0]//seq*seq]
        self._ids = _ids.reshape(-1, seq)

    def __len__(self):
        return self._ids.shape[0]

    def __getitem__(self, index):
        return self._ids[index]
    
class SftDataset(Dataset):
    def  __init__(self, file_path, max_seq_len):
        self._max_seq_len=max_seq_len
        with open(file_path,"rb") as fr:
            self.datas=pickle.load(fr)
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, index):
        '''
        _prompt,_tag都是列表
        '''
        _prompt,_tag=self.datas[index]
        _prompt_len=len(_prompt)
        if _prompt_len<=self._max_seq_len:
            _fill_zero=(self._max_seq_len-_prompt_len)*[0,]
            _prompt=_prompt+_fill_zero
            _tag=_tag+_fill_zero
        else:
            _prompt=_prompt[:self._max_seq_len]
            _tag=_tag[:self._max_seq_len]
        return (torch.tensor(_prompt,dtype=torch.long)),(torch.tensor(_tag,dtype=torch.long))
        

if __name__ == "__main__":
    _dataset = PretrainDataset("datas/2022-21_zh_middle_0011", 10)
    
