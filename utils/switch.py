import dataloaders
import helpers
import models

def get_modules(args):
  config = args
  dataloader = getattr(dataloaders, config.dataloader)
  helper = getattr(helpers, config.helper)
  model = getattr(models, config.model)

  output = (dataloader, model, helper)
  return output

def get_train_aux(args, model):
  optimizers = helpers.optimizer(args, model)
  optimizer = getattr(optimizers, args.optimizer)

  lr_schedulers = helpers.lr_scheduler(args, optimizer)
  lr_scheduler = getattr(lr_schedulers, args.lr_scheduler)

  return optimizer, lr_scheduler