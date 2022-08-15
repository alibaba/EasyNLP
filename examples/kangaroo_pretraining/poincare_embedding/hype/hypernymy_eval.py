#!/usr/bin/env python3

from . import build_model
from hypernymysuite.base import HypernymySuiteModel
from hypernymysuite.evaluation import all_evaluations
import torch as th
import os
from subprocess import check_call


def download_data():
    data_dir = os.environ.get('HYPERNYMY_DATA_DIR', 'data')
    if not os.path.exists(os.path.join(data_dir, 'bless.tsv')):
        print('Downloading hypernymysuite eval data...')
        url = 'https://raw.githubusercontent.com/facebookresearch/hypernymysuite/master/download_data.sh'  # noqa B950
        env = {'HYPERNYMY_DATA_OUTPUT': data_dir}
        res = check_call(f'wget -q -O - {url} | bash', shell=True, env=env)
        if res != 0:
            raise ValueError('')


class EntailmentConeModel(HypernymySuiteModel):
    def __init__(self, chkpnt, **kwargs):
        self.model = build_model(chkpnt['conf'], len(chkpnt['model']['lt.weight']))
        self.model.load_state_dict(chkpnt['model'])
        self.vocab = {w : i for i, w in enumerate(chkpnt['objects'])}
        self.vocab['<OOV>'] = 0

    def idx(self, w):
        return self.vocab[w] if w in self.vocab else 0

    def predict_many(self, hypo, hyper, ans=None):
        device = self.model.lt.weight.device

        # Get embeddings
        hypo_t = th.LongTensor([self.idx(h) for h in hypo]).to(device)
        hyper_t = th.LongTensor([self.idx(h) for h in hyper]).to(device)
        hypo_e = self.model.lt(hypo_t)
        hyper_e = self.model.lt(hyper_t)

        # Compute entailment cone energy
        dists = self.model.energy(hypo_e, hyper_e)

        # words are not hypernyms of themselves
        dists[hypo_t == hyper_t] = 1e10
        return -dists.cpu().numpy()


def main(chkpnt, cpu=False):
    download_data()
    extra_args = {'map_location' : 'cpu'} if cpu else {}
    if isinstance(chkpnt, str):
        assert os.path.exists(chkpnt)
        chkpnt = th.load(chkpnt, **extra_args)

    model = EntailmentConeModel(chkpnt)

    # perform the evaluations
    with th.no_grad():
        results = all_evaluations(model)

    results['epoch'] = chkpnt['epoch']

    def iter(d, res, path, sum, count):
        if isinstance(d, dict):
            for k in d.keys():
                sum, count = iter(d[k], res, path + '_' + k, sum, count)
        elif 'val_inv' in path and 'ap100' not in path:
            res[path[1:]] = d  # strip leading tab
            sum += d
            count += 1
        return sum, count
    summary = {}
    sum, count = iter(results, summary, '', 0, 0)
    summary['eval_hypernymy_avg'] = sum / count
    return results, summary
