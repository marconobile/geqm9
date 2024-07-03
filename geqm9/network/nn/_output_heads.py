from typing import List, Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro import Linear
from geqtrain.nn.allegro._fc import ScalarMLPFunction
# from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from torch_scatter import scatter


@compile_mode("script")
class Head(GraphModuleMixin, torch.nn.Module):

    '''
    input shape [B,N,k]
    '''

    def __init__(
        self,
        field: str,
        out_irreps: Union[o3.Irreps, str],
        irreps_in: dict[str, o3.Irreps] = {}, # for super ctor call, taken automatically from prev module
        out_field: Optional[str] = None, # on which key of the AtomicDataDict to place output
        head_function=ScalarMLPFunction,
        head_function_kwargs={},
    ):
        super().__init__()
        self.field = field
        irreps = irreps_in[field]
        self.out_field = out_field
        self.head_function = head_function

        # here take irreps_in of the data that u want to use i.e. field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps},
            irreps_out={out_field: out_irreps},
        )

        irreps_muls = []
        n_l = {}
        n_dim = 0
        for mul, ir in irreps:
            irreps_muls.append(mul)
            n_l[ir.l] = n_l.get(ir.l, 0) + 1
            n_dim += ir.dim
        assert all([irreps_mul == irreps_muls[0] for irreps_mul in irreps_muls])

        self.irreps_mul = irreps_muls[0]
        self.n_l = n_l

        out_irreps = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)
        assert all([l == 0 for l in out_irreps.ls])

        self.head = head_function(
                mlp_input_dimension=self.irreps_mul * self.n_l[0],
                mlp_output_dimension=out_irreps.dim, # .dim takes number of entries in vect
                **head_function_kwargs,
            )

        self.dropout = torch.nn.Dropout(.2)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # add batch dim if not present

        data = AtomicDataDict.with_batch(data)

        # get a single feature vector for g by summing node features

        features = data[self.field]
        graph_feature = scatter(features, data[AtomicDataDict.BATCH_KEY], dim=0)

        # dropout

        graph_feature = self.dropout(graph_feature)

        # pass thru head

        data[self.out_field] = self.head(graph_feature)

        return data


