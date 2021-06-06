import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.metrics import Loss, MetricsLambda
from ignite.contrib.handlers import ProgressBar

from transformers import AutoTokenizer, AutoConfig

import logging
from pprint import pformat
import argparse
import json
import math
from pathlib import Path
from functools import partial
from copy import deepcopy

from utils.switch import get_modules
from utils.auxiliary import set_seed
from utils.argument import verify_args

logger = logging.getLogger(__file__)


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  # TODO: copy params to checkpoints
  parser.add_argument("--params_file", type=str, help="JSON configuration file")
  # parser.add_argument("--generate_config", type=str, help="JSON configuration file to generate")
  parser.add_argument("--dataset_path", type=str, default="data", help="Path of the dataset.")
  parser.add_argument("--dataset_cache", type=str, default='data/dataset_cache', help="Path of the dataset cache")
  parser.add_argument("--model_checkpoint", type=str, required=True, help="Path, url or short name of the model")
  parser.add_argument("--score_file", type=str, required=True, help="Score file Path")
  parser.add_argument("--record_name", type=str, required=True, help="Result Name")
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

  dataloader, models, helper = get_modules(args)

  logger.info("Arguments: %s", pformat(args))

  args.n_gpu = 1
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args.device = device

  # Set seed
  set_seed(args)

  # Model construction
  config = AutoConfig.from_pretrained(args.model_name_or_path, return_dict=True)
  SPECIAL_TOKENS = dataloader.SPECIAL_TOKENS
  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  tokenizer.add_special_tokens(SPECIAL_TOKENS)
  args._tokenizer = tokenizer

  dataset_class = getattr(dataloader, args.dataloader_class)
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

  # Load checkpoint
  checkpoint_fp = Path(args.model_checkpoint)
  if checkpoint_fp.is_dir():
    checkpoint_fp = max(filter(lambda x: x.name.startswith("best_model"), checkpoint_fp.iterdir()), key=lambda x: float(x.stem.split('=')[-1]))
  assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
  logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
  checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
  Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

  if args.debug:
    setattr(dataset_class, "__len__", lambda _: 10)

  # Prepare metrics
  model.eval()
  _inference = partial(helper.evaluator_update, args=args, model=model)
  evaluator = Engine(_inference)
  if not hasattr(helper, "evalutor_metrics"):
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][0], x[1][0])),}
    metrics["ppl"] = MetricsLambda(math.exp, metrics["nll"])
  else:
    metrics = helper.evalutor_metrics(args)
  for name, metric in metrics.items():
    metric.attach(evaluator, name)

  # Display
  pbar = ProgressBar(persist=True)
  pbar.attach(evaluator)

  @evaluator.on(Events.COMPLETED)
  def record_results():
    pbar.log_message("Validation on test set: %s" % pformat(evaluator.state.metrics))
    result_file = Path(args.score_file)
    if result_file.exists():
      with result_file.open() as fr:
        old_results = json.load(fr)
    else:
      old_results = {}

    if args.record_name in old_results: # if ppl run behind, merge result
      old_results[args.record_name].update(evaluator.state.metrics)
    else:  # else add key
      old_results[args.record_name] = evaluator.state.metrics

    with result_file.open('w') as fw:
      json.dump(old_results, fw, indent=2)

  evaluator.run(test_loader)

if __name__ == '__main__':
  main()
