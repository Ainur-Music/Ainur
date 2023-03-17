import torch
from data.dataset import get_dataset
from torch.utils.data import DataLoader
from model.clip import CLIP
import torch.nn.functional as F

dataset = get_dataset("/Users/gio/spotdl/", crop=2**20)
loader = DataLoader(dataset, batch_size=3, shuffle=True)
clip = CLIP()

for batch in loader:
    audio, *_, lyrics = batch
    loss = clip(audio, lyrics)
    print(loss)
    print(clip.encode_audio(audio).shape)
    print(clip.encode_lyrics(lyrics).shape)