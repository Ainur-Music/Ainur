import torch
from archisound import ArchiSound

def get_autoencoder(name="dmae1d-ATC32-v3"):
    return ArchiSound.from_pretrained(name)


if __name__ == "__main__":
    autoencoder = get_autoencoder()
    # Compression 32x
    # Downsampling 512x
    x = torch.randn(1, 2, 2**20) # [1, 2, t]
    z = autoencoder.encode(x) # [1, 32, t/2^9]
    y = autoencoder.decode(z, num_steps=1) # [1, 2, t]

    print(f"x shape: {x.shape}")
    print(f"z shape: {z.shape}")
    print(f"y shape: {y.shape}")