import torch
import os
import json
import logging

from itertools import chain
from utils.data import truncate_sequences, pad_ids, pad_ids_2D
from dataloaders.datawalker import DatasetWalker

from tqdm import tqdm

logger = logging.getLogger(__name__)

# additional_special_tokens = ["</ksp>"]

SPECIAL_TOKENS = {
  "bos_token": "<|startoftext|>",
  "eos_token": "<|endoftext|>",
  "pad_token": "<|padoftext|>",
  # "additional_special_tokens": additional_special_tokens,
}

class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, args, tokenizer, split_type, labels=True):
    self.args = args
    self.dataroot = args.dataroot
    self.tokenizer = tokenizer
    self.split_type = split_type

    self.bos = self.tokenizer.bos_token_id
    self.eos = self.tokenizer.eos_token_id
    self.pad = self.tokenizer.pad_token_id

    self.all_response_tokenized = []
    self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, debug=args.debug)
    self._create_examples()
    self.all_response_tokenized = list(map(eval, set(map(str, self.all_response_tokenized))))


  def _create_examples(self):
    logger.info("Creating examples")
    self.examples = []
    for dialog in tqdm(self.dataset_walker.data, disable=self.args.local_rank not in [-1, 0]):
      dialog_id = dialog["dialog_id"]
      history = dialog["history"]
      response = dialog["response"]

      tokenized_history = self.tokenizer.encode(history, add_special_tokens=False)
      truncated_tokenized_history = tokenized_history[ - self.args.history_max_tokens:]

      tokenized_response = self.tokenizer.encode(response, add_special_tokens=False)
      self.all_response_tokenized.append(tokenized_response)

      self.examples.append({
        "history": truncated_tokenized_history,
        "response": tokenized_response,
        "response_text": response,
        "dialog_id": dialog_id,
      })
  
  def build_input_from_segments(self, history, response, with_eos=True):
    """ Build a sequence of input from 2 segments: history and last reply """
    instance = {}

    sequence = history + [self.eos] + response + ([self.eos] if with_eos else [])

    instance["input_ids"] = sequence
    instance["input_mask"] = [1, ] * len(instance["input_ids"])

    # build response
    instance["lm_labels"] = [-100] * len(history) + [-100] + response + [self.eos] 
  
    return instance, sequence

  def __getitem__(self, index):
    example = self.examples[index]
    instance, sequence = self.build_input_from_segments(
      example["history"],
      example["response"]
    )
    return instance

  def __len__(self):
    return len(self.examples)

  def collate_fn(self, batch):
    input_ids = [ins["input_ids"] for ins in batch]
    input_mask = [ins["input_mask"] for ins in batch]
    lm_labels = [ins["lm_labels"] for ins in batch]

    input_ids = torch.tensor(pad_ids(input_ids, self.pad))
    input_mask = torch.tensor(pad_ids(input_mask, 0))
    lm_labels = torch.tensor(pad_ids(lm_labels, -100))

    return input_ids, input_mask, lm_labels

class testDataset(BaseDataset):
  def __getitem__(self, index):
    example = self.examples[index]
    instance, sequence = self.build_input_from_segments(
      example["history"],
      [],
      with_eos=False
    )
    return instance, example

  def collate_fn(self, batch):
    input_ids = [ins["input_ids"] for ins, _ in batch]
    input_mask = [ins["input_mask"] for ins, _ in batch]
    dialog_id = [example["dialog_id"] for _, example in batch]
    history = [example["history"] for _, example in batch]
    response = [example["response_text"] for _, example in batch]

    input_ids = torch.tensor(pad_ids(input_ids, self.pad, direction='left'))
    input_mask = torch.tensor(pad_ids(input_mask, 0, direction='left'))

    return input_ids, input_mask, (dialog_id, history, response)
