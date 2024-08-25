import torch.fft
import torch
import numpy as np

def AnomalyGenerator(latent, mean, std, wf, wt):
    # 转换到频域
    latent_freq = torch.fft.fft(latent)

    # 添加噪声
    gaussian_noise_f = torch.tensor(np.random.normal(mean, std, size=latent_freq.shape).astype("float32"),
                                    device=latent.device)
    latent_fake_freq = latent_freq + wf * gaussian_noise_f

    # 转换回时域
    latent_fake = torch.fft.ifft(latent_fake_freq).real

    gaussian_noise_t = torch.tensor(np.random.normal(mean, std, size=latent_freq.shape).astype("float32"),
                                 device=latent.device)
    latent_fake = latent_fake + wt * gaussian_noise_t

    return latent_fake


    # # 添加噪声
    # gaussian_noise = torch.tensor(np.random.normal(mean, std, size=latent.shape).astype("float32"),
    #                                 device=latent.device)
    # latent_fake = latent + gaussian_noise
    # return latent_fake

    # # 转换到频域
    # latent_freq = torch.fft.fft(latent)
    #
    # # 添加噪声
    # gaussian_noise_f = torch.tensor(np.random.normal(mean, std, size=latent_freq.shape).astype("float32"),
    #                                 device=latent.device)
    # latent_fake_freq = latent_freq + gaussian_noise_f
    #
    # # 转换回时域
    # latent_fake = torch.fft.ifft(latent_fake_freq).real
    #
    # return latent_fake
