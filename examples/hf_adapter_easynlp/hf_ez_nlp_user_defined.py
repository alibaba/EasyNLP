import torch
import torch.nn.functional as F


def compute_loss(self, forward_outputs, label_ids):
        return {'loss': F.cross_entropy(forward_outputs[1], label_ids, ignore_index=-100, reduction='mean')}

def forward_repre(self, model, args, batch):
    batch = {
        key: val.to(args.local_rank) if isinstance(val, torch.Tensor) else val
        for key, val in batch.items()
    }
    label_ids=None
    if "label_ids" in batch.keys():
        label_ids = batch.pop("label_ids")
    forward_outputs = model(batch["input_ids"],batch["attention_mask"],batch["token_type_ids"],labels=label_ids)
    return forward_outputs, label_ids, batch