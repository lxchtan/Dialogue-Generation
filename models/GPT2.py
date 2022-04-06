from transformers.models.gpt2 import GPT2LMHeadModel

class GenerationModel(GPT2LMHeadModel):
  def __init__(self, config, args=None):
    super().__init__(config)