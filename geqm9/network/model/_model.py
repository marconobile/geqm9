from typing import Optional
import logging
from geqm9.utils import DataDict

from e3nn import o3
from geqtrain.data import AtomicDataDict, AtomicDataset
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
)

from geqm9.network.nn import (
    Head,
    OutputScaler,
    ScalarsMultiHeadSelfAttention
)

from geqtrain.nn import (
    OneHotAtomEncoding,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    ReadoutModule,
    NodewiseReduce,
)


def Model(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    logging.debug("Building model")

    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config.get("parity", "o3_full")
        assert parity_setting in ("o3_full", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        # check consistency
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        config["irreps_edge_sh"] = irreps_edge_sh

    layers = {
        # -- Encode --
        "node_attrs": (
            EmbeddingNodeAttrs,
            dict(
                embedding_dim = config['node_embedding_dims'],
            )
        ),
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
    }

    layers.update(
        {
            "interaction": (
            InteractionModule,
                dict(
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY, # 1-hot atom types; shape ([N, n_atom_types])
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY, # radial embedding of displacement vectors: BESSEL(8) enc; shape: ([num_edges, 8]) 8= number of bessel for encoding
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY, # angular embedding of displacement vectors: SH enc Lmax=2; shape: ([num_edges, 9]) (eg 9 if Lmax=2)
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY, # "edge_features", these are only scalars!
                    output_hidden_irreps=True, # this defines the fact that this layer does not outputs the requested out_irreps
                    # as requested in yaml but outs an hidden vector -> a repr for each edge
                ),
            ),
            "per_node_features": (
                EdgewiseReduce, # takes all edges outgoing for node i and sum their features to get 1 feat vect of the node i itself
                dict(
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_FEATURES_KEY, # so this creates a feat vect for each node
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "output_head": (
                Head,
                dict(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=DataDict.GRAPH_OUTPUT_KEY,
                ),
            ),
            "output_scaler": (
                OutputScaler,
                dict(
                    out_field=DataDict.GRAPH_OUTPUT_KEY,
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
