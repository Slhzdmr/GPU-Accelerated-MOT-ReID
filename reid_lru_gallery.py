import torch
from collections import OrderedDict

class LRUReIDGallery:
    def __init__(self,max_identities=50,similarity_threshold=0.85,device="cuda"):
        self.max_identities = max_identities
        self.threshold = similarity_threshold
        self.device = device
        self.embeddings = OrderedDict()

    def register_or_update(self, identity_id, embedding):
        if identity_id in self.embeddings:
            self.embeddings.pop(identity_id)

        self.embeddings[identity_id] = embedding

        if len(self.embeddings) > self.max_identities:
            self.embeddings.popitem(last=False)

    def match_identity(self,current_embedding):
        if not self.embeddings:
            return None, 0.0

        gallery_ids = list(self.embeddings.keys())

        gallery_matrix = torch.stack(list(self.embeddings.values()))
        scores = torch.mv(gallery_matrix,current_embedding)

        best_score, best_idx = torch.max(scores,dim=0)
        best_score_val = best_score.item()

        if best_score_val >= self.threshold:
            best_match_id = gallery_ids[best_idx.item()]
            self.register_or_update(best_match_id,current_embedding)
            return best_match_id, best_score_val
        return None, best_score_val
