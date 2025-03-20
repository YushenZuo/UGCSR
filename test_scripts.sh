CUDA_VISIBLE_DEVICES=1 python test_ugcsr.py \
    --pretrained_model_path /home/vip2024/zys/Agentic-Restoration/executor/super_resolution/tools/OSEDiff/pretrained_models/SD/stable-diffusion-2-1-base \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --pretrained_fusion_path pretrained_models/fusion/fusion_model.pth \
    --fusion_tile 1024 \
    --fusion_tile_overlap 64 \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/synthetic_val_LR/LR \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions/PISASR_pretrained_wavelet_align_fusion/synthetic



CUDA_VISIBLE_DEVICES=0 python test_ugcsr.py \
    --pretrained_model_path /home/vip2024/zys/Agentic-Restoration/executor/super_resolution/tools/OSEDiff/pretrained_models/SD/stable-diffusion-2-1-base \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --pretrained_fusion_path pretrained_models/fusion/fusion_model.pth \
    --fusion_tile 1024 \
    --fusion_tile_overlap 64 \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/wild_val \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions/test_wild