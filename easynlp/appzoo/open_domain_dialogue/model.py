from ..application import Application
from easynlp.modelzoo import TransformerConfig, TransformerModel
import torch

class OpenDomainDialogue(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        self.config = TransformerConfig()
        self.backbone = TransformerModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.backbone.NULL_IDX, reduction='none'
        )
    
    def forward(self, inputs):
        # logits: bsz * output_len * vocab_size
        # preds: bsz * output_len
        outputs = self.backbone(**inputs)
        logits, preds, hidden_states = outputs
        return {
            "hidden": hidden_states,
            "logits": logits,
            "predictions": preds,
            "probabilities": torch.softmax(logits, dim=-1)
        }

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        # logits_view: (bsz * output_len) * vocab_size
        # label_ids: bsz * output_len
        logits = forward_outputs['logits']
        logits_view = logits.reshape(-1, logits.size(-1))
        loss = self.criterion(logits_view, label_ids.view(-1))
        return {"loss": loss}
    
    def _generate(self, input, beam_size, max_ts):
        model_input = input['text_vec'].unsqueeze(0)
        self.backbone.eval()
        return self.backbone._generate(model_input, beam_size, max_ts)