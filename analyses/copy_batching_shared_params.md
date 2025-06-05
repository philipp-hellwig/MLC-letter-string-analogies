|                 | 0                                                   |
|:----------------|:----------------------------------------------------|
| dataset         | data/all_transformations_study1_incl_copy_fixed_gen |
| batch_size      | 32                                                  |
| query_first     | False                                               |
| nepochs         | 20                                                  |
| lr              | 0.001                                               |
| lr_end_factor   | 0.05                                                |
| lr_warmup       | True                                                |
| hidden_size     | 128                                                 |
| input_size      | 30                                                  |
| output_size     | 30                                                  |
| PAD_idx_input   | 28                                                  |
| PAD_idx_output  | 28                                                  |
| nlayers_encoder | 3                                                   |
| nlayers_decoder | 3                                                   |
| nhead           | 8                                                   |
| dropout_p       | 0.1                                                 |
| ff_mult         | 4                                                   |
| activation      | gelu                                                |