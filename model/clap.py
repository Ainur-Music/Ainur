import torch
from torchvision.transforms import ToPILImage
from autoencoder import get_autoencoder
from transformers import ClapModel, ClapConfig, AutoProcessor


def get_clip():
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")

    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    return model, processor
    
if __name__ == "__main__":
    model, processor = get_clip()

    audio_autoencoder = get_autoencoder()

    lyrics = """
    But the form of a life is long never-ending 

    And the smell of your halo, I know 
    And the smile of a knife is seldom befriending 
    And the smell of tangelo, I know

    When I'm near you, I feel like a king 
    A life force inside to do anything 
    When I'm downtown, I pick up the phone 
    I hear you and I'm not alone

    And the smell of your halo, I know 
    And the smile of a knife is seldom befriending 
    And the smell of tangelo, I know
    """
    text = ["Hypnotic Raggae Song Red Hot Chili Peppers", "Beatles happy country"]
    audio = torch.randn(2**18)


    with torch.no_grad():
        inputs = processor(text=text, audios=[audio, audio], return_tensors="pt", padding=True, sampling_rate=48_000)
        outputs = model(**inputs)
        logits_per_audio = outputs.logits_per_audio  # this is the image-text similarity score
        probs = logits_per_audio.softmax(dim=1)
    print(probs)