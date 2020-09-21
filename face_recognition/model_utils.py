from face_recognition.model import Backbone
import torch
import numpy

class face_verifier():
    def __init__(self, net_depth=50, drop_ratio=0.6, net_mode="ir_se", device="cuda"):
        # create model
        self.model = Backbone(net_depth, drop_ratio, net_mode).to(device)
        save_path = "face_recognition/model_ir_se50.pth"
        # load model
        self.model.load_state_dict(torch.load(save_path))
        self.model.eval()

    def verify_person(self, f1, f2):
        # 0: same / 1: ambiguous / 2: different
        batch_tensor = torch.cat([f1, f2], 0)
        output_feat = self.model(batch_tensor.cuda())
        sim = torch.nn.CosineSimilarity(dim=0)
        sim = sim(output_feat[0], output_feat[1]).data.cpu().numpy()
        if sim > 0.7: # same
            return 0
        elif sim > 0.5: # ambiguous
            return 1
        else:
            return 2
