## NNCF ONNX backend

### `nncf.compress_weights()` method

List of supported parameters for `nncf.compress_weights()` method when the `model` type is `onnx.ModelProto`

| Parameter           | Status              | Note |
|---------------------|:-------------------:|------|
| model               | Supported           | |
| mode                | Partially Supported | Not supported modes: NF4, MXFP4, MXFP8_E4M3, FP8_E4M3, FP4, NVFP4, CODEBOOK, ADAPTIVE_CODEBOOK, CB4|
| ratio               | Supported           | |
| group_size          | Supported           | |
| ignored_scope       | Supported           | |
| all_layers          | Supported           | |
| dataset             | Supported           | |
| sensitivity_metric  | Supported           | |
| subset_size         | Supported           | |
| awq                 | Supported           | |
| scale_estimation    | Supported           | |
| gptq                | Not Supported       | |
| lora_correction     | Not Supported       | |
| backup_mode         | Supported           | |
| compression_format  | Supported           | |
| advanced_parameters | Partially Supported | `AdvancedCompressionParameters.statistics_path` not supported |

### `nncf.quantize()` method

List of supported parameters for `nncf.quantize()` method when the `model` type is `onnx.ModelProto`

| Parameter            | Status              | Note |
|----------------------|:-------------------:|------|
| model                | Supported           | |
| calibration_dataset  | Supported           | |
| mode                 | Not Supported       | |
| preset               | Supported           | |
| target_device        | Partially Supported | `CPU_SPR` not supported |
| subset_size          | Supported           | |
| fast_bias_correction | Supported           | |
| model_type           | Supported           | |
| ignored_scope        | Supported           | |
| advanced_parameters  | Supported           | |
