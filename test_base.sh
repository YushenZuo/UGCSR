CUDA_VISIBLE_DEVICES=1 python test_ugcsr_base.py \
    --pretrained_model_path /home/vip2024/zys/Agentic-Restoration/executor/super_resolution/tools/OSEDiff/pretrained_models/SD/stable-diffusion-2-1-base \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/synthetic_test_LR/LR \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions_test/PISASR_pretrained_wavelet_align_fusion_base/synthetic



CUDA_VISIBLE_DEVICES=1 python test_ugcsr_base.py \
    --pretrained_model_path /home/vip2024/zys/Agentic-Restoration/executor/super_resolution/tools/OSEDiff/pretrained_models/SD/stable-diffusion-2-1-base \
    --pretrained_path pretrained_models/pisa_sr.pkl \
    --input_image /home/data1/NTIRE2025/ShortformUGCSR/wild_test \
    --output_dir /home/data1/NTIRE2025/ShortformUGCSR/submissions_test/PISASR_pretrained_wavelet_align_fusion_base/wild