import torch
import torch.nn as nn

class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__() # super가 뭐였더라?
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict
        
    @property # property가 뭐야?
    def device(self):
        return next(iter(self.parameters())).device #<- next하고 iter 함수가 뭐임?
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs): # 함수 안의 함수가 있음요
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return
            
            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)
            
        def load_dict(state_dict, prefix): # prefix가 뭘까?
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #      import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict
        
        self.params_dict = load_dict(state_dict, prefix +'params_dict')
        self.params_dict.requires_grad_(False)
        return
                    