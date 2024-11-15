import torch
import sentencepiece as spm
from model import Skyer
from torch.distributions.categorical import Categorical


class Eval:

    def __init__(self,
                 topk=4,
                 temp=2):

        self._topk = topk
        self._temp = temp

        self._skyer = Skyer(num_layers=2,
                            input_dim=128,
                            hide_dim=96,
                            n_q_heads=2,
                            n_kv_heads=1,
                            max_len=1024,
                            num_vocs=30000,
                            cache_max_batch_size=1,
                            cache_max_seq_len=1024).cuda()
        self._skyer.eval()
        self._skyer.load_state_dict(torch.load("/root/workspace/LMT/myllm_infer/sft_save/sft_0/mp_rank_00_model_states.pt"),strict=False)

        self._spm = spm.SentencePieceProcessor()
        self._spm.Load("/root/workspace/LMT/myllm_infer/tokenizer.model")

    def __call__(self, prompt):
        _vocs = prompt
        _prompt_ids = self._spm.Encode(prompt)
        _ids = torch.tensor(_prompt_ids, dtype=torch.long)[None].cuda()
        '''
        _ids为SV结构
        
        '''
        _id, _voc = self.forward(_ids, 0)
        _vocs += _voc
        _start_pos = _ids.shape[1]

        for _ in range(100):
            _id, _voc = self.forward(_id, _start_pos)
            _start_pos += 1
            _vocs += _voc
        return _vocs

    def forward(self, ids, start_pos):
        _os = self._skyer(ids, start_pos)
        '''
        _os是NSV结构
        _o是NV结构
        top-k
        T-softmax
        概率取样
        '''
        _o = _os[:, -1]
        _weight, _indices = torch.topk(_o, self._topk, dim=-1)
        _probs = self._tsoftmax(_weight, self._temp)
        # _m = Categorical(_probs)
        # _s = _m.sample()
        _s = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim=-1, index=_s)
        return _id, self._spm.Decode(_id.item())

    @staticmethod
    def _tsoftmax(xs, temp=1.):
        _o = xs-xs.mean()
        return torch.exp(_o/temp)/(torch.exp(_o/temp).sum(-1)+1e-5)


if __name__ == '__main__':

    env = Eval()

    voc = env("只剩一个心脏还能活吗？")
    print(voc)
