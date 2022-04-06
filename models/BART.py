from transformers.models.bart.modeling_bart import BartForConditionalGeneration

class GenerationModel(BartForConditionalGeneration):
  def __init__(self, config, args=None):
    super().__init__(config)