import torch

_max_seq_len=10
_prompt=[1,2,3,4]
print(_prompt)
_prompt_len=len(_prompt)
_fill_zero = (_max_seq_len-_prompt_len)*[0,]
print(_fill_zero)
_prompt = _prompt + _fill_zero
print(_prompt)