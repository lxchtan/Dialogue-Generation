import os
import json

class DatasetWalker(object):
  """
    history: [str, ]
    label: {'response': str, 'emotion': str (single word)}
  """
  def __init__(self, dataset, dataroot, labels, debug=False):
    self.labels = labels

    path = os.path.join(os.path.abspath(dataroot))

    if dataset not in ['train', 'valid', 'test']:
      raise ValueError('Wrong dataset name: %s' % (dataset))

    file = os.path.join(path, f'{dataset}.json')
    with open(file, 'r') as f:
      examples = json.load(f)

    # Reduce the time of loading dataset when debugging
    if debug:
      examples = examples[:20]

    self.data = []
    assert dataset in ['train', 'valid', 'test']

    for example in examples:
      new_example = {}
      
      new_example["dialog_id"] = example["dialog_id"]
      new_example["history"] = example["post"]
      new_example["response"] = example["response"]

      self.data.append(new_example)