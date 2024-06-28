from typing import Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro import Linear
from geqtrain.nn.allegro._fc import ScalarMLPFunction
# from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from torch_scatter import scatter

def exists(val):
    return val is not None


@compile_mode("script")
class OutputScaler(GraphModuleMixin, torch.nn.Module):
    def __init__(self,
                out_field: str,
                out_irreps: Union[o3.Irreps, str],
                irreps_in: dict[str, o3.Irreps] = {}, # for super ctor call, taken automatically from prev module
                target_means: list = None,
                target_stds : list = None,
                learnable: bool = True,
                **kwargs, # catch all input form yaml
                ):
        super().__init__()

        self.out_field = out_field
        irreps = irreps_in[out_field]

        self._init_irreps(
            irreps_in = irreps_in,
            my_irreps_in ={out_field: irreps},
            irreps_out = {out_field: out_irreps}
        )

        if learnable:
            self.means = torch.nn.Parameter(torch.as_tensor(target_means)) if target_means else None
            # self.stds  = torch.nn.Parameter(torch.as_tensor(target_stds))  if target_stds  else None
        else:
            self.means = self.register_buffer("means", torch.as_tensor(target_means)) if target_means else None
            # self.stds  = self.register_buffer("stds",  torch.as_tensor(target_stds))  if target_stds  else None

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # if self.stds is not None:
        #     data[self.out_field] *= self.stds

        # if self.means is not None:
        #     data[self.out_field] += self.means

        data[self.out_field] = torch.exp(data[self.out_field]) # map into [0, inf]

        return data