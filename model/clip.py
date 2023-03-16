import torch
from torchvision.transforms import ToPILImage
from autoencoder import get_autoencoder
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from utils import MelSpectrogram


def get_clip(max_lenght=512):
    configuration = CLIPConfig()
    configuration.text_config.max_position_embeddings = max_lenght

    # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    model = CLIPModel(configuration)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
    text = "Hypnotic Raggae Song Red Hot Chili Peppers"
    spec = MelSpectrogram()
    audio = torch.randn(2, 2**18)
    image = ToPILImage()(spec(audio))

    inputs = processor(text=[lyrics], images=[image, image], return_tensors="pt", padding=True, max_length=512, truncation=True)

    outputs = model(**inputs)
    print(outputs['text_embeds'].shape)
    print(outputs['image_embeds'].shape)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
