from transformers import AdamW
from ignite.contrib.handlers import PiecewiseLinear, ParamGroupScheduler


class optimizer(object):
  def __init__(self, args, model) -> None:
    self.args = args
    self.model = model

  @property
  def AdamW(self):
    return AdamW(self.model.parameters(), lr=self.args.lr)

  @property
  def group_tm(self):
    all_params = self.model.parameters()
    z_params = list(self.model.z_model.parameters())
    params_id = list(map(id, z_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    opt = AdamW([
        {'params': other_params},
        {'params': z_params, 'lr': self.args.lr}
    ], lr=self.args.lr * 5)
    return opt


class lr_scheduler(object):
  def __init__(self, args, optimizer) -> None:
    self.args = args
    self.optimizer = optimizer

  @property
  def normal(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (t_total, 0.0)])

    return lr_scheduler

  @property
  def multi_step_1(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr),        (steps_per_epoch, 0.0),
                                                     (steps_per_epoch+1, args.lr),   (t_total, 0.0)])

    return lr_scheduler

  @property
  def multi_step(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr),        (steps_per_epoch * 3, 0.0),
                                                     (steps_per_epoch*3+1, args.lr),   (t_total, 0.0)])
    return lr_scheduler

  @property
  def group_tm(self):
    args = self.args
    optimizer = self.optimizer
    steps_per_epoch = args.steps_per_epoch
    t_total = steps_per_epoch * args.n_epochs
    scheduler1 = PiecewiseLinear(optimizer, 'lr', [(0, args.lr), (steps_per_epoch * 3, 0.0), (steps_per_epoch*3+1, args.lr*5), (t_total, 0.0)], param_group_index=0)
    scheduler2 = PiecewiseLinear(optimizer, 'lr', [(0, args.lr), (steps_per_epoch * 3, 0.0), (steps_per_epoch*3+1, args.lr), (t_total, 0.0)], param_group_index=1)
    lr_schedulers = [scheduler1, scheduler2]
    names = ["lr (base)", "lr (z_model)"]
    scheduler = ParamGroupScheduler(schedulers=lr_schedulers, names=names)
    return scheduler