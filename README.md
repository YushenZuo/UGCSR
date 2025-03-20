
<div align="center">
<h2>Code for NTIRE 2025 Challenge on Short-form UGC Image Super-Resolution (4x)</h2>

</div>


## ⏰ Key
Combine PiSA-SR and NAFSSR.



## 🍭 Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2.1-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

#### Step 2: Running testing command 
Example script:
```
CUDA_VISIBLE_DEVICES=1 python test_ugcsr.py \
    --pretrained_model_path /home/vip2024/zys/Agentic-Restoration/executor/super_resolution/tools/OSEDiff/pretrained_models/SD/stable-diffusion-2-1-base \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --pretrained_fusion_path pretrained_models/fusion/fusion_model.pth \
    --fusion_tile 1024 \
    --fusion_tile_overlap 64 \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/wild_val \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_align_fusion/wild
```



### License
This project is released under the [Apache 2.0 license](LICENSE).

