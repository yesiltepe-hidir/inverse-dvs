VIDEO_PATH="data/corgi_beach.mp4"

python i2v_ddim_inversion.py \
    --model_path "THUDM/CogVideoX-5b-I2V" \
    --lora_path "None" \
    --prompt "" \
    --video_path $VIDEO_PATH \
    --output_path "inversions" \
    --guidance_scale 1.0 \
    --subfolder "reconstruction"

# k=0 is default ddim inversion. 
# k=1 is first order k-rnr-diffusion. 
# k=2 is second order k-diffusion. best performance in non-adaptive k-rnr.
K_ORDER=0
python reconstruct.py \
    --model_path "THUDM/CogVideoX-5b-I2V" \
    --lora_path "None" \
    --guidance_scale 1.0 \
    --k_order $K_ORDER \
    --prompt "a corgi in the beach" \
    --video_path $VIDEO_PATH \
    --inverted_latent_path "inversions/reconstruction_latents_data_30.pt" \
    --output_path "output"