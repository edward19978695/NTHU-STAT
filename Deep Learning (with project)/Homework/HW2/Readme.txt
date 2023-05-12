1. model.py : 建構並訓練 AE 和 VAE 模型，將其各自表現最佳的 model weight 存成 best_model_cnn.pth 和 best_model_vae.pth
2. performance.py : 導入 best_model_cnn.pth 和 best_model_vae.pth 並計算模型 output 的表現（若助教想要確認 PSNR/SSIM score 可直接執行此檔案）
3. noise.py : 導入 best_model_cnn.pth 和 best_model_vae.pth 並計算模型 latent tensor 再加上 Gaussian noise 後計算模型 output，結果存成 gen_data.npy 和 gen_label.npy