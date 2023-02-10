# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BeamSearch classes for BlenderBot """

from typing import Optional, List, TypeVar, Iterable, Tuple, Set, Dict
from typing_extensions import TypedDict
import torch
import math
from abc import abstractmethod
from operator import attrgetter

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

TSType = TypeVar('TSType', bound='TreeSearch')

def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


class SearchBlocklist(object):
    """
    Search block list facilitates blocking ngrams from being generated.
    """

    def __init__(self) -> None:
        self._phrases: Set[str] = set()
        self._phrase_ngrams: Dict[int, List[List[int]]] = {}

    def __bool__(self):
        return bool(self._phrases)

    def clear(self) -> None:
        self._phrases = set()
        self._phrase_ngrams = {}

    def items(self) -> Iterable[Tuple[int, List[List[int]]]]:
        return self._phrase_ngrams.items()

class _HypothesisTail(object):
    """
    Hold some bookkeeping about a hypothesis.
    """

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid', 'token_details']

    def __init__(self, timestep, hypid, score, tokenid, token_details):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid
        self.token_details = token_details

class _PathSelectionTokenDetails(TypedDict, total=False):
    token_logprob: float  # conditional log-probability of token (normalized)
    token_rank: int  # rank of token in conditional distribution

class _PathSelection(object):
    """
    Output of TreeSearch:select_paths.

    Represents output of path selection process.
    """

    __slots__ = ['hypothesis_ids', 'token_ids', 'scores', 'token_details']

    def __init__(
        self,
        hypothesis_ids,
        token_ids,
        scores,
        token_details: Optional[List[_PathSelectionTokenDetails]] = None,
    ):
        self.hypothesis_ids = hypothesis_ids
        self.token_ids = token_ids
        self.scores = scores
        self.token_details = token_details  # length equal to beam size

class TreeSearch(object):
    """
    Abstract Tree Search class.

    It keeps information about beam_size concurrent, developing hypotheses. Concrete
    implementations make choices about which token to explore next at each point in the
    tree. Different choices result in different generation algorithms.
    """

    def __init__(
        self,
        beam_size,
        block_ngram=-1,
        context_block_ngram=-1,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_length=3,
        device='cpu',
        length_penalty=0.65,
        verbose=False,
        gpu_beam_blocking=False,
    ):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param block_ngram:
            size of ngrams to block.
        :param context_block_ngram:
            size of context ngrams to block
        :param padding_token:
            padding token ID
        :param bos_token:
            beginning of sentence token ID
        :param eos_token:
            end of sentence token ID
        :param min_length:
            minimum length of the predicted sequence
        :param device:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.block_ngram = block_ngram
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.context = None
        self.context_block_ngram = context_block_ngram
        self.block_list: Optional[SearchBlocklist] = None
        self.device = device
        # recent score for each hypo in the beam
        self.scores = None
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]

        self.verbose = verbose
        # (beam size, sample length) list of lists containing token-level data for each token in each hypo in the beam
        self.token_details: Optional[List[List[_PathSelectionTokenDetails]]] = None
        if self.verbose:
            self.token_details = []
            for _ in range(self.beam_size):
                self.token_details.append([{"token_logprob": 0.0, "token_rank": 0}])

        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.gpu_beam_blocking = gpu_beam_blocking
        self.partial_hyps = torch.tensor([[self.bos] for i in range(beam_size)])
        if self.gpu_beam_blocking:
            self.partial_hyps = self.partial_hyps.cuda()

    def set_context(self: TSType, context: torch.LongTensor) -> TSType:
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context = context.tolist()
        return self

    def set_batch_context(
        self: TSType,
        batch_context_list: torch.LongTensor,
        batch_idx: int,
        gpu_beam_blocking: bool,
    ) -> TSType:
        """
        Version of .set_context() that operates on a single element of a batch.

        Set the internal context representation and return self.

        :param batch_context_list:
            a list of lists, each one containing the context for one member of the batch
        :param batch_idx:
            index of the batch
        :param gpu_beam_blocking:
            whether we are using gpu kernel for beam blocking, if so return a tensor,
            else return a list.
        """
        context = batch_context_list[batch_idx]
        self.context = context if gpu_beam_blocking else context.tolist()
        return self

    def set_block_list(self: TSType, block_list: Optional[SearchBlocklist]) -> TSType:
        self.block_list = block_list
        return self

    def get_output_from_current_step(self):
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    @abstractmethod
    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a {hypothesis_ids, token_ids, scores, token_details} , where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
            - token_details is a (beamsize) list of objects with with metadata about each generated token.
        """
        pass

    def _block_ngrams(
        self,
        ngram_size: int,
        logprobs: torch.Tensor,
        step: int = 0,
        if_context_blocking=False,
    ):
        """
        Hard block ngrams from the logprobs.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param step:
            current step on generating utterances
        :param if_context_blocking:
            whether we are doing context blocking
        """
        # cpu beam blocking
        for beam_id, hyp in enumerate(self.partial_hyps.tolist()):
            if len(hyp) < ngram_size - 1:
                continue
            source = hyp if if_context_blocking is False else self.context
            prefix = hyp[-(ngram_size - 1) :]
            for i in range(len(source) - ngram_size + 1):
                ngram = source[i : i + ngram_size]
                if ngram_size == 1 or prefix == ngram[:-1]:
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
        if self.block_list is None:
            return logprobs

        for beam_id, hyp in enumerate(self.partial_hyps.tolist()):
            for ngram_size, bad_ngrams in self.block_list.items():
                prefix = hyp[-(ngram_size - 1) :]
                for ngram in bad_ngrams:
                    if (ngram_size == 1) or prefix == ngram[:-1]:
                        logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def advance(self, logprobs, step):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = neginf(logprobs.dtype)

        if self.scores is None:
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)

        # penalize hypotheses ending in EOS on the prior scores (self.scores) level
        # this is related to search which uses prior scores (self.scores) (e.g. beam)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = neginf(self.scores.dtype)

        # beam blocking
        if self.block_ngram > 0:
            # self blocking
            logprobs = self._block_ngrams(
                ngram_size=self.block_ngram,
                logprobs=logprobs,
                step=step,
                if_context_blocking=False,
            )

        logprobs = self._block_block_list(logprobs)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError(
                    "Must use TreeSearch.set_context to use context blocking."
                )
            # context blocking
            logprobs = self._block_ngrams(
                ngram_size=self.context_block_ngram,
                logprobs=logprobs,
                step=step,
                if_context_blocking=True,
            )

        path_selection = self.select_paths(logprobs, self.scores, current_length)
        self.scores = path_selection.scores
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(path_selection.token_ids)
        self.bookkeep.append(path_selection.hypothesis_ids)

        # this checking for device seems suboptimal
        # might need to change later
        if self.partial_hyps.get_device() == -1:
            hyp_device = 'cpu'
        else:
            hyp_device = self.partial_hyps.get_device()
        self.partial_hyps = torch.cat(
            (
                self.partial_hyps[path_selection.hypothesis_ids.long()],
                path_selection.token_ids.view(path_selection.token_ids.shape[0], -1).to(
                    hyp_device
                ),
            ),
            1,
        )

        if self.verbose:
            assert path_selection.token_details
            assert self.token_details
            for i in range(self.beam_size):
                self.token_details[i].append(path_selection.token_details[i])

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= neginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished

                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
                    token_details=self.token_details[hypid][-1]
                    if self.token_details is not None
                    else None,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def is_done(self):
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.beam_size

    def _find_ngrams(self, input_list, n):
        """
        Find ngrams of size n in input list.
        """
        return list(zip(*[input_list[i:] for i in range(n)]))

    def _get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs) - 1

        :param hyp_id:
            id with range up to beam_size - 1

        :return:
            hypothesis sequence
        """
        hyp_idx = []
        endback = hypothesis_tail.hypid

        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                _HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                    token_details=self.token_details[endback][i]
                    if self.token_details is not None
                    else None,
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails):
        """
        Return hypothesis as a tensor of token ids.
        """
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses according to adjusted scores.

        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.

        :param n_best:
            number of finalized hypotheses to return

        :return:
            list of (tokens, score, token_metadata) 3-tuples, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
              - token_metadata dictionary:
                    token_logprobs -> a tensor of conditional log probabilities of tokens
                    token_ranks -> a tensor of ranks of tokens in vocabulator, by probability, when sampled
        """
        # if we never actually finished, force one
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(
                _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.outputs[-1][0],
                    token_details=self.token_details[0][-1]
                    if self.token_details is not None
                    else None,
                )
            )

        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            rescored_finished.append(
                _HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                    token_details=finished_item.token_details,
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        n_best_list = []
        for hyp in srted:
            hyp_data = self._get_hyp_from_finished(hyp)
            token_ids = self._get_pretty_hypothesis(hyp_data)
            token_metadata = (
                [tok.token_details for tok in reversed(hyp_data)]
                if self.verbose
                else None
            )
            n_best_list.append((token_ids, hyp.score, token_metadata))

        # check that there is at least one finished candidate
        # and assert that each of them contains only one EOS
        assert (
            len(n_best_list) >= 1
        ), f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for (pred, score, _) in n_best_list:
            assert (pred == self.eos).sum() == 1, (
                f'TreeSearch returned a finalized hypo with multiple end tokens '
                f'with score {score.item():.2f}'
            )

        return n_best_list
    
class BeamSearch(TreeSearch):
    """
    Beam search.
    """

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.
        """
        # if numel is 1, then this is the first time step, only one hyp is expanded
        if prior_scores.numel() == 1:
            logprobs = logprobs[0:1]

        # beam search actually looks over all hypotheses together so we flatten
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = torch.div(best_idxs, voc_size, rounding_mode='trunc')
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            probs = torch.softmax(logprobs, dim=-1)
            tok_probs = (
                torch.index_select(probs, 0, hyp_ids)
                .gather(1, tok_ids.unsqueeze(1))
                .view(-1)
            )
            tok_ranks = (
                probs.argsort(1, descending=True)
                .argsort(1)
                .view(-1)
                .gather(0, best_idxs)
            )

            token_details = []

            for tok_logprob, tok_rank in zip(
                tok_probs.log().cpu().numpy(), tok_ranks.cpu().numpy()
            ):
                token_details.append(
                    {
                        "token_logprob": tok_logprob.item(),
                        "token_rank": int(tok_rank.item()),
                    }
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )