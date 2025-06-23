| filename_model                           | batching_method   | query_first   |   loss |   accuracy in-dist |   accuracy out-of-dist |
|:-----------------------------------------|:------------------|:--------------|-------:|-------------------:|-----------------------:|
| MLC_batchunstruct_dallstudy1_nep20.pt    | unstructured      | False         |  1.079 |              0.68  |                  0.118 |
| MLC_batchbytrans_dallstudy1_nep20.pt     | transformation    | False         |  1.203 |              0.64  |                  0.121 |
| MLC_batchbyalph_dallstudy1_nep20.pt      | alphabet          | False         |  1.112 |              0.618 |                  0.084 |
| MLC_batchbyboth_dallstudy1_nep20.pt      | both              | False         |  1.257 |              0.602 |                  0.098 |