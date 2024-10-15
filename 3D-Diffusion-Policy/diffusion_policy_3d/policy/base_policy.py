from typing import Dict
import torch
import torch.nn as nn

from diffusion_policy_3d.model.commom.module_attr_mixin import ModuleAttrMixin
from diffusion_policy_3d.model.commom.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return B, Ta, Da
        """
        raise NotImplementedError() #NotImplementedError 이게 뭐야?
    
    # reset state for stateful polices
    def reset(self):
        pass
    
    # =========== training =============
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()