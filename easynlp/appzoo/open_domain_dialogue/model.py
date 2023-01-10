from ..application import Application
from easynlp.modelzoo import TransformerConfig, TransformerModel

class OpenDomainDialogue(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        self.config = TransformerConfig()
        self.backbone = TransformerModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
    
    def _generate(self, input, beam_size, max_ts):
        model_input = input['text_vec'].unsqueeze(0)
        return self.backbone._generate(model_input, beam_size, max_ts)