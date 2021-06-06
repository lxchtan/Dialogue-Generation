# import torch
import re
import os
import sys
import json
import nltk
import argparse
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from utils import embedding_metrics
from collections import OrderedDict

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def distinct(seqs):
  """ Calculate intra/inter distinct 1/2. """
  intra_dist1, intra_dist2 = [], []
  unigrams_all, bigrams_all = Counter(), Counter()

  for seq_line in seqs:
    seq = nltk.word_tokenize(seq_line.strip())

    unigrams = Counter(seq)
    bigrams = Counter(zip(seq, seq[1:]))
    intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
    intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

    unigrams_all.update(unigrams)
    bigrams_all.update(bigrams)

  inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
  inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
  intra_dist1 = np.average(intra_dist1)
  intra_dist2 = np.average(intra_dist2)
  return intra_dist1, intra_dist2, inter_dist1, inter_dist2


class Metric:
  def __init__(self, args):
    self.reset()
    self.args = args

  def reset(self):
    self._generation_bleu1 = 0.0
    self._generation_bleu2 = 0.0
    self._generation_bleu3 = 0.0
    self._generation_bleu4 = 0.0
    self._generation_meteor = 0.0
    self._generation_rouge_1 = 0.0
    self._generation_rouge_2 = 0.0
    self._generation_rouge_l = 0.0
    self._generation_bertscore_P = 0.0
    self._generation_bertscore_R = 0.0
    self._generation_bertscore_F = 0.0
    self._generation_bertscore_hashname = ''
    self.refs = []
    self.hyps = []

  def _match(self, ref_knowledge, pred_knowledge):
    result = []
    for pred in pred_knowledge:
      matched = False
      for ref in ref_knowledge:
        if pred['domain'] == ref['domain'] and pred['entity_id'] == ref['entity_id'] and pred['doc_id'] == ref['doc_id']:
          matched = True
      result.append(matched)
    return result

  def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k=5):
    relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

    if True in relevance:
      idx = relevance.index(True)
      result = 1.0/(idx+1)
    else:
      result = 0.0

    return result

  def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
    relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

    if True in relevance:
      result = 1.0
    else:
      result = 0.0

    return result

  def _normalize_text(self, text):
    result = text.lower()
    result = RE_PUNC.sub(' ', result)
    result = RE_ART.sub(' ', result)
    result = ' '.join(result.split())

    return result

  def _bleu(self, ref_response, hyp_response, n=4):
    ref_tokens = self._normalize_text(ref_response).split()
    hyp_tokens = self._normalize_text(hyp_response).split()

    weights = [1.0/n] * n

    chencherry = SmoothingFunction()
    score = sentence_bleu([ref_tokens], hyp_tokens, weights, chencherry.method1)

    return score

  def _meteor(self, ref_response, hyp_response):
    score = single_meteor_score(ref_response, hyp_response, self._normalize_text)

    return score

  def _rouge(self, ref_response, hyp_response, mode='l'):
    ref_response = self._normalize_text(ref_response)
    hyp_response = self._normalize_text(hyp_response)

    rouge = Rouge()

    if mode == 'l':
      score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-l']['f']
    elif mode == 1:
      score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-1']['f']
    elif mode == 2:
      score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-2']['f']
    else:
      raise ValueError("unsupported mode: %s" % mode)

    return score

  def cal_bertscore(self):
    import bert_score
    (P, R, F) = \
        bert_score.score(self.hyps, self.refs, lang="en")
    self._generation_bertscore_P = P.sum().item()
    self._generation_bertscore_R = R.sum().item()
    self._generation_bertscore_F = F.sum().item()

  def update(self, ref_obj, hyp_obj):
      self._generation_bleu1 += self._bleu(ref_obj, hyp_obj, 1)
      self._generation_bleu2 += self._bleu(ref_obj, hyp_obj, 2)
      self._generation_bleu3 += self._bleu(ref_obj, hyp_obj, 3)
      self._generation_bleu4 += self._bleu(ref_obj, hyp_obj, 4)
      self._generation_meteor += self._meteor(ref_obj, hyp_obj)
      self._generation_rouge_l += self._rouge(ref_obj, hyp_obj, 'l')
      self._generation_rouge_1 += self._rouge(ref_obj, hyp_obj, 1)
      self._generation_rouge_2 += self._rouge(ref_obj, hyp_obj, 2)

      # self.refs.append(self._normalize_text(ref_obj.strip()))
      # self.hyps.append(self._normalize_text(hyp_obj.strip()))
      self.refs.append(ref_obj.strip())
      self.hyps.append(hyp_obj.strip())

  def _compute(self, score_sum):
    return score_sum / len(self.refs)

  def scores(self):

    self.cal_bertscore()

    generation_bleu1_f = self._compute(self._generation_bleu1)
    generation_bleu2_f = self._compute(self._generation_bleu2)
    generation_bleu3_f = self._compute(self._generation_bleu3)
    generation_bleu4_f = self._compute(self._generation_bleu4)
    generation_meteor_f = self._compute(self._generation_meteor)
    generation_rouge_l_f = self._compute(self._generation_rouge_l)
    generation_rouge_1_f = self._compute(self._generation_rouge_1)
    generation_rouge_2_f = self._compute(self._generation_rouge_2)
    generation_bertscore_P_f = self._compute(self._generation_bertscore_P)
    generation_bertscore_R_f = self._compute(self._generation_bertscore_R)
    generation_bertscore_F_f = self._compute(self._generation_bertscore_F)

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(self.hyps)

    EAS, GMS, ES = embedding_metrics.cal_all(self.refs, self.hyps, self.args.embeddings)

    scores = {
        'bleu-1': generation_bleu1_f,
        'bleu-2': generation_bleu2_f,
        'bleu-3': generation_bleu3_f,
        'bleu-4': generation_bleu4_f,
        'meteor': generation_meteor_f,
        'rouge_1': generation_rouge_1_f,
        'rouge_2': generation_rouge_2_f,
        'rouge_l': generation_rouge_l_f,
        'bertscore_P': generation_bertscore_P_f,
        'inter_dist1': inter_dist1,
        'inter_dist2': inter_dist2,
        "Embedding Average Score": EAS,
        "Greedy Matching Score": GMS,
        "Extrema Score": ES
    }

    return scores


def main():
  parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

  parser.add_argument('--result_file', action='store', required=True,
                      help='File containing output')
  parser.add_argument('--scorefile', action='store', required=True,
                      help='File containing scores')
  parser.add_argument('--embeddings', help="embeddings bin file, recommend to download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM")

  args = parser.parse_args()

  # load the result_file
  with open(args.result_file, 'r') as f:
    result_data = json.load(f)
    refs = []
    preds = []
    for ex in result_data:
      refs.append(ex['ground_truth'])
      preds.append(ex['generated'])
  assert len(preds) == len(refs)
  
  metric = Metric(args)

  for ref, pred in tqdm(zip(refs, preds), total=len(preds)):
    metric.update(ref, pred)
    # try:
    #   metric.update(ref, pred)
    # except:
    #   print(ref)
    #   print(pred)
    #   print("="*10)
    #   pass

  scores = metric.scores()
  # print(scores)
  result_dicts = OrderedDict({
    args.result_file: scores
  })

  if os.path.exists(args.scorefile):
    with open(args.scorefile) as fin:
      old_results = json.load(fin)
      # if ppl run first, merge result
      if args.result_file in old_results:
        result_dicts[args.result_file].update(old_results[args.result_file])
      result_dicts.update(old_results)

  with open(args.scorefile, 'w') as fw:
    json.dump(result_dicts, fw, indent=2)

  # with open(args.scorefile, 'a') as out:
  #   out.write(args.result_file + '\n')
  #   json.dump(scores, out, indent=2)
  #   out.write('\n')

if __name__ == "__main__":
  main()
