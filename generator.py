import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.utils import setup_logger

from transformers import AdamW, AutoTokenizer, AutoConfig

import logging
from pprint import pformat
import argparse
import os
import json
import math
from pathlib import Path
from functools import partial
from copy import deepcopy

from utils.switch import get_modules
from utils.auxiliary import set_seed, average_distributed_scalar
from utils.argument import verify_args, update_additional_params, set_default_params, set_default_dataset_params

from tqdm import tqdm

logger = logging.getLogger(__file__)


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  # TODO: copy params to checkpoints
  parser.add_argument("--params_file", type=str, help="JSON configuration file")
  parser.add_argument("--generate_config", type=str, help="JSON configuration file to generate")
  parser.add_argument("--dataset_path", type=str, default="data", help="Path of the dataset.")
  parser.add_argument("--dataset_cache", type=str, default='data/dataset_cache', help="Path of the dataset cache")
  parser.add_argument("--model_checkpoint", type=str, required=True, help="Path, url or short name of the model")
  parser.add_argument("--result_file", type=str, required=True, help="Path generate result")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
  parser.add_argument("--share_encoder", action='store_true', help="Share the parameters between two encoders.") 
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument("--fp16", type=str, default="",
                      help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
  parser.add_argument("--seed", type=int, default=43)
  parser.add_argument("--debug", action='store_true')
  args = parser.parse_args()

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )

  verify_args(args, parser)

  # load args from params file and update the args Namespace
  with open(args.params_file, "r") as f:
    params = json.load(f)
    args = vars(args)
    args.update(params)
    args = argparse.Namespace(**args)
  
  with open(args.generate_config, "r") as f:
    params = json.load(f)
    args = vars(args)
    args.update(params)
    args = argparse.Namespace(**args)

  dataloader, models, helper = get_modules(args)

  logger.info("Arguments: %s", pformat(args))

  args.n_gpu = 1
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args.device = device

  # Set seed
  set_seed(args)

  # Model construction
  config = AutoConfig.from_pretrained(args.model_name_or_path)
  SPECIAL_TOKENS = dataloader.SPECIAL_TOKENS
  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  tokenizer.add_special_tokens(SPECIAL_TOKENS)
  args._tokenizer = tokenizer

  dataset_class = getattr(dataloader, "testDataset") # TODO: suitable
  test_dataset = dataset_class(args, tokenizer, 'test')
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn) #, collate_fn=test_dataset.collate_fn
  
  # Model construction
  model_class = getattr(models, args.model_class)
  if hasattr(args, "decoder_model_name_or_path"):
    from models.base_modify import EncoderDecoderConfig
    config_encoder = deepcopy(config)
    config_knowledge_encoder = deepcopy(config)
    config_decoder = deepcopy(config)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_knowledge_encoder, config_decoder)
    
  model = model_class(config=config, args=args)
  model.resize_token_embeddings(len(tokenizer))
  model = model.to(device)

  checkpoint_fp = Path(args.model_checkpoint)
  if checkpoint_fp.is_dir():
    checkpoint_fp = max(filter(lambda x: x.name.startswith("best_model"), checkpoint_fp.iterdir()), key=lambda x: float(x.stem.split('=')[-1]))
  assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
  logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
  checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
  Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

  if args.debug:
    setattr(dataset_class, "__len__", lambda _: 10)

  model.eval()
  all_output_texts = []
  run_batch_generation_sample = helper.greedy_sample
  for did, batch in enumerate(tqdm(test_loader, desc="Generating", disable=args.debug)):
    with torch.no_grad():
      sampled_output_ids, ground_truth, dialog_id, history = run_batch_generation_sample(batch, args, model, test_dataset)
      sampled_output_text = tokenizer.batch_decode(sampled_output_ids, skip_special_tokens=True)

      for didt, gt, res, his in zip(dialog_id, ground_truth, sampled_output_text, history):
        example = {}
        example["dialog_id"] = didt
        example["ground_truth"] = gt
        example["generated"] = res
        all_output_texts.append(example)

        if args.debug:
          print(f"Dialog: {didt}")
          print(tokenizer.decode(his, skip_special_tokens=True))
          print("Generate:", res)
          print("  Ground:", gt)
          print()

  if not os.path.exists("results"):
    os.mkdir("results")
  with open(os.path.join("results", args.result_file), "w") as fout:
    json.dump(all_output_texts, fout, indent=2)

if __name__ == '__main__':
  main()
