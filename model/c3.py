import torch
from torchmetrics import Metric
import torch.nn.functional as F

class C3(Metric):
    def __init__(self, clasp):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clasp = clasp
        self.add_state("similarity", torch.tensor(0.).to(device), dist_reduce_fx="sum")
        self.add_state("num", torch.tensor(0).to(device), dist_reduce_fx="sum")

    def update(self, preds, target):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = preds.shape[0]
        msg = f"preds shape ({preds.shape}) does not match target shape ({len(target)}) at dimension 0"
        assert preds.shape[0] == len(target), msg

        preds_emb = self.clasp.encode_audio(preds).to(device)
        target_emb = self.clasp.encode_lyrics(target).to(device)

        self.similarity += torch.sum(F.cosine_similarity(preds_emb, target_emb))
        self.num += batch_size

    def compute(self):
        assert self.num != 0, "[C3] Batch dimension zero."
        return self.similarity / self.num