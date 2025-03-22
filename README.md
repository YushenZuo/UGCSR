
<div align="center">
<h2>PiNAFusion-SR: Code for NTIRE 2025 Challenge on Short-form UGC Image Super-Resolution (4x)</h2>

</div>


## ⏰ Key
PiNAFusion-SR: Combine PiSA-SR with image fusion model, which is inspired by NAFSSR.



## 🍭 Quick Inference
#### Step 1: Set Up
```
conda create -n ugcsr python=3.10
conda activate ugcsr
pip install -r requirements.txt
```

#### Step 2: Download the pretrained models
- Download the pretrained SD-2.1-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

#### Step 3: Running testing command 
Example script:
```bash
CUDA_VISIBLE_DEVICES=1 python test_ugcsr.py \
    --pretrained_model_path <The-folder-you-save-SD-2.1-Base> \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --pretrained_fusion_path pretrained_models/fusion/fusion_model.pth \
    --fusion_tile 1024 \
    --fusion_tile_overlap 64 \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/wild_val \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_align_fusion/wild
```



### License
This project is released under the [Apache 2.0 license](LICENSE).

