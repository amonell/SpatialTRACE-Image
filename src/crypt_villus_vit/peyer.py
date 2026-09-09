"""Released shared-scale, representation-pretrained Peyer image classifier."""
import torch
from torch import nn
from .model import MultitaskDapiVit


class PeyerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.encoder_architecture != 'shared_scale_aware' or not config.retain_scale_embeddings
                or config.local_readout != 'center_4x' or config.context_readout != 'center_4x'
                or not all((config.use_local_branch, config.use_context_branch, config.use_fine_branch))
                or config.fusion_architecture != 'concat'):
            raise ValueError('Peyer classifier requires the released shared-scale axis architecture')
        base = MultitaskDapiVit(config)
        self.config = config
        self.encoder_architecture = config.encoder_architecture
        self.shared_encoder = base.shared_encoder
        self.fine_encoder = base.fine_encoder
        self.head = base.head
        self.classifier = base.axis_head

    @property
    def axis_head(self):
        return self.classifier

    def forward(self, batch):
        parts = []
        for scale, branch in enumerate(('local', 'context')):
            image = batch[branch + '_image']
            ids = torch.full((len(image),), scale, dtype=torch.long, device=image.device)
            parts.append(self.shared_encoder.encode_image(image, ids, readout='center_4x'))
        parts.append(self.fine_encoder(batch['fine_image']))
        logit = self.classifier(self.head(torch.cat(parts, dim=1))).squeeze(1)
        return {'axis_logit': logit, 'predicted_axis_coordinate': logit.sigmoid()}
