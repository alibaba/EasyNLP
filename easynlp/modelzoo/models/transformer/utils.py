from typing import List, Any, Dict, Tuple, Generator, Optional, TypeVar, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import os
from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as torch_checkpoint
import weakref
import logging
import itertools

@dataclass
class ThreadLocalCheckpointingState(threading.local):
    is_checkpointing: bool = False
    is_recomputing: bool = False
    is_checkpointing_disabled: bool = False


thread_local = ThreadLocalCheckpointingState()

class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling :func:`unpack_non_tensors`.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        dummy_tensor_requires_grad: torch.Tensor,
        run_function: Any,
        parent_ctx_dict: Dict[str, Any],
        kwarg_keys: Tuple[str, ...],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        torch_checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()
        ctx.had_autocast_in_fwd = is_autocast_enabled()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(x.to("cpu", non_blocking=True) for x in tensor_inputs)
        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad(), enable_checkpointing():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)
            the_module = unpacked_args[0]

        # Because we run with torch.no_grad(), we can't actually access
        # outputs.requires_grad. Instead, we manually compute it by
        # checking if either the input or the module needs grads
        parameters = list(the_module.parameters())

        # If the module is wrapped by FlattenParamsWrapper, then the
        # parameters would have been deleted. If so, we need to access
        # the views into the flattened parameters.
        if hasattr(the_module, "_unflattened_param_views"):
            parameters += the_module._unflattened_param_views

        output_requires_grad = any(param.requires_grad for param in parameters) or any(
            x.requires_grad for x in tensor_inputs
        )
        parent_ctx_dict["output_requires_grad"] = output_requires_grad

        if not isinstance(outputs, torch.Tensor):
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs

        return outputs

def pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)

    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return tuple(kwarg_keys), tuple(flat_args)

def unpack_kwargs(kwarg_keys: Tuple[str, ...], flat_args: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs

def unpack_non_tensors(
    tensors: Tuple[torch.Tensor, ...], packed_non_tensors: Optional[Dict[str, List[Any]]]
) -> Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict), type(packed_non_tensors)
    mixed: List[Any] = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list), (
        f"len(tensors) {len(tensors)} len(objects) {len(objects)} " f"len(is_tensor_list) {len(is_tensor_list)}"
    )
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)

def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any, ...]]
) -> Tuple[Tuple[torch.Tensor, ...], Optional[Dict[str, List[Any]]]]:
    """
    Split a tuple into a list of tensors and the rest with information
    for later reconstruction.

    When called with a tensor X, will return: (x,), None

    Usage::

        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        assert tensors == (x, y)
        assert packed_non_tensors == {
            "is_tensor": [True, True, False, False],
            "objects": [None, 3],
        }
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors: List[torch.Tensor] = []
    packed_non_tensors: Dict[str, List[Any]] = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors
    
def get_rng_state() -> Dict[str, Any]:
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state

def is_autocast_enabled() -> bool:
    """Similar to torch.is_autocast_enabled, but compatible with torch 1.5.1"""
    if hasattr(torch, "is_autocast_enabled"):
        return torch.is_autocast_enabled()
    return False

@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig

def checkpoint_wrapper(
    module: nn.Module,
    offload_to_cpu: bool = False,
) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.

    In terms of GPU memory savings:

        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.

    ..Note::

        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.

    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # Patch the batchnorm layers in case there are any in this module.
    patch_batchnorm(module)

    # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
    # When such cycle exists, gc won't collect the module when the module is freed.
    # That causes GPU memory to be leaked. See the unit test for how we catch that.
    #
    # We prefer this over a class wrapper since the class wrapper would have to
    # proxy a lot of fields and methods.
    module.forward = functools.partial(  # type: ignore
        _checkpointed_forward, type(module).forward, weakref.ref(module), offload_to_cpu
    )
    return module

def patch_batchnorm(module: nn.Module) -> List:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.

       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().

    Args:
        module (nn.Module):
            The module to be patched in-place.

    Returns:
        (list):
            A list of hook handles, late can be freed.
    """

    def pre_forward(module: _BatchNorm, input: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module._track_running_stats_backup = module.track_running_stats
        module.track_running_stats = False

    def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module.track_running_stats = module._track_running_stats_backup

    hooks = []
    for name, child in module.named_modules():
        # _BatchNorm is base for bn1d, bn2d, bn3d and sync_bn, apex_sync_bn, etc.
        if isinstance(child, _BatchNorm) and not hasattr(child, "disable_patch_batchnorm"):
            # Register the pre/post hooks.
            pre_handle = child.register_forward_pre_hook(pre_forward)
            post_handle = child.register_forward_hook(post_forward)
            hooks += [pre_handle, post_handle]
    return hooks

def _checkpointed_forward(
    original_forward: Any, weak_self: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any
) -> Any:
    module = weak_self()

    # If gradients are disabled, just use original `.forward()` method directly.
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)

    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {
        "offload": offload_to_cpu,
    }
    # Dummy tensor with grad is used to ensure the backward pass is called. This is needed
    # when original_forward's input are non-tensor (i.e. a tuple). Using this dummy tensor
    # avoids requiring users to set their input tensors's requires_grad flag. In the case
    # of tuple type inputs, setting the flag won't even trigger the backward pass.
    #
    # One implication of this is that since we always feed in a dummy tensor
    # needing grad, then the output will always require grad, even if it originally
    # wouldn't, such as if the module and original input both do not require grad.
    # We get around this by saving the desired requires_grad value in output and
    # detaching the output if needed.
    output = CheckpointFunction.apply(
        torch.tensor([], requires_grad=True), original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    output_requires_grad = parent_ctx_dict["output_requires_grad"]
    if not isinstance(output, torch.Tensor):
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        output = [x.detach() if not output_requires_grad else x for x in output]

        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)

    else:
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        if not output_requires_grad:
            output = output.detach()

    return output



"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def default(val, default):
    """
    shorthand for explicit None check for optional arguments.
    """
    return val if val is not None else default

_seen_logs: Set[str] = set()

def warn_once(msg: str) -> None:
    """
    Log a warning, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.warning(msg)

def trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in the model.

    :param model:
        the model whose parameters we wish to count.

    :return:
        total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

Chunk = TypeVar('Chunk')

PipelineWorkItem = namedtuple(
    'PipelineWorkItem', ['chunk_idx', 'layer_nos', 'next_device']
)
    
class PipelineHelper(object):
    """
    PipelineHelper assists with implementing pipelining in model parallelism.

    For a tutorial on model parallelism, as it's implemented in parts of ParlAI,
    see https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html.

    Usage:
    >>> my_model = PipelineHelper().make_parallel(my_model)

    Note that you will need to manually implement logic which handles the
    moved layers.
    """

    def __init__(self):
        self.__device_allocations = {}
        self.num_devices = torch.cuda.device_count()
        self.devices = []
        for i in range(self.num_devices):
            d = f'cuda:{i}'
            self.devices.append(d)
            self.__device_allocations[d] = 0

    def check_compatibility(self, opt):
        """
        Check compatibility for opts.

        Really just used to raise an error message if the user mixes multiprocessing and
        model parallelism.
        """
        if opt.get('multiprocessing') and not os.environ.get('PARLAI_FORCE_MP'):
            raise RuntimeError(
                "It looks like you are trying to mix multiprocessing data "
                "parallelism (multiprocessing_train or multiprocessing_eval) "
                "with --model-parallel true. This is almost certainly a user "
                "error, and is going to result in hanging as the two methods "
                "fight for resources. Use simple `train_model` instead of "
                "`mp_train`, or add `--model-parallel false`. For more info, "
                "see https://github.com/facebookresearch/ParlAI/issues/2962."
            )

    def make_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Allocate specific layers in a model to be ModelParallel.

        Limited to only ModuleLists within the model.  Uses some heuristics to
        attempt to evenly distribute layers across GPUs, in order to balance
        memory usage. They are:

        - Assume the 0th GPU will host the optimizer, word embeddings, etc.
        - Assume activation memory is linear with the number of parameters.
        - All layers are approximately equal in size.
        """

        # first assume all layers will go onto gpu 0 as an optimizer. The
        # optimizer and embeddings are not quite as expensive as the
        # activations (which scale via batchsize), Empirically, I found this
        # heuristic works well enough. The weighting factor of 3 is more or
        # less made up.
        self.__device_allocations['cuda:0'] += trainable_parameters(model) * 3

        model.apply(self._place_modulelist)
        model._apply(self._move_rest_to_cuda0)  # type: ignore
        return model

    def _move_rest_to_cuda0(self, parameter: torch.Tensor):
        if parameter.device.type == 'cpu':
            return parameter.to('cuda:0')
        else:
            return parameter

    def _place_modulelist(self, submodule: torch.nn.Module) -> None:
        if not isinstance(submodule, torch.nn.ModuleList):
            # not a ModuleList, leave it untouched
            return
        if getattr(submodule, 'model_parallel_exempt', False):
            return

        assert isinstance(submodule, torch.nn.ModuleList)  # for typechecker
        layers = submodule

        # mark this section as MP
        layers.is_model_parallel = True  # type: ignore

        # next, let's figure out how many parameters we can assign to each GPU,
        # but not make actual assignments yet. Assignments come later because we
        # want consecutive layers to be collocated
        keyfunc = self.__device_allocations.__getitem__
        layer_assignments = {k: 0 for k in self.devices}
        for layer_no, layer in enumerate(layers):
            if layer_no == 0:
                # hard code the first layer to be 0.
                mostfree = 'cuda:0'
            else:
                # otherwise dynamic allocation
                mostfree = min(self.devices, key=keyfunc)
            # 32 is a totally arbitrary, made up number that worked in practice
            # on the large models I tested on. I believe it should be roughly
            # batch size, but this was set empirically.
            self.__device_allocations[mostfree] += trainable_parameters(layer) * 32
            # mark a layer as going to the given element
            layer_assignments[mostfree] += 1

        devices = [d for i, d in enumerate(self.devices[:]) if layer_assignments[d] > 0]
        for layer_no, layer in enumerate(layers):
            layer_gpu = devices[0]
            assert layer_assignments[layer_gpu] > 0
            logging.debug(f"Model Parallel: Assigning {layer_no} to {layer_gpu}")
            layer._mp_gpu = layer_gpu
            layers[layer_no] = layer.to(layer_gpu)
            layer_assignments[layer_gpu] -= 1
            if layer_assignments[layer_gpu] == 0:
                devices.pop(0)

    @staticmethod
    def guess_split_size(item: Chunk, num_gpus: Optional[int] = None, dim=0) -> int:
        """
        Estimate the number of chunks we should split the batch into via heuristics.
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()  # type: ignore

        if isinstance(item, torch.Tensor):
            if num_gpus == 1:
                # no point in chunking if we're not really doing model parallel
                return item.size(dim)
            # heuristic: use the same number of chunks as 2 * num_gpus.  this
            # isn't perfect (it ideally would be tuned differently for every model
            # and number of GPUs), but it seems to work reasonably wellenough in several
            # architectures tested.
            return max(1, item.size(dim) // int(num_gpus * 2))
        elif isinstance(item, tuple):
            return PipelineHelper.guess_split_size(item[0], num_gpus)
        elif isinstance(item, dict):
            return PipelineHelper.guess_split_size(list(item.values())[0], num_gpus)
        raise TypeError(f'Cannot determine split size for {type(item)}')

    @staticmethod
    def split(item: Chunk, split_size: Optional[int] = None, dim=0) -> List[Chunk]:
        """
        Split a tensor or group of tensors into smaller chunks of the same type.

        :param item:
            The item being split. May be a Tensor, a tuple of Tensors, or a
            dictionary mapping str -> Tensor.
        :param split_size:
            The maximum size of each output chunk. If None, we will guess using
            heuristics
        :param dim:
            The dimension to split along.
        """
        if split_size is None:
            split_size = PipelineHelper.guess_split_size(item)

        if isinstance(item, torch.Tensor):
            # base case, just split the tensor
            return list(torch.split(item, split_size, dim))
        elif isinstance(item, tuple):
            # We start with Tuple[Tensor] and we return List[Tuple[Tensor]]
            return list(zip(*(PipelineHelper.split(i, split_size, dim) for i in item)))
        elif isinstance(item, dict):
            if item == {}:
                # Terrible edge case: the empty dict. We handle by returning an
                # infinite list of empty dicts and we'll figure out its correct
                # size later. This happens for the incremental_state in
                # MultiheadAttention.
                return itertools.repeat({})  # type: ignore

            # we can't handle dicts with empty objects in them, due to how we handle
            # the case above.  awkward syntax because pytorch 1.3 doesn't like
            # comparing tensors to dicts.
            if {} in [x for x in item.values() if isinstance(x, dict)]:
                raise ValueError(
                    'Cannot handle a dictionary with an empty dictionary inside.'
                )
            if () in [x for x in item.values() if isinstance(x, tuple)]:
                raise ValueError(
                    'Cannot handle a dictionary with an empty tuple inside.'
                )

            # we start with Dict[key,tensor]
            # we map it to d: Dict[key, List[Tensor]], where we have split each mapping
            d = {k: PipelineHelper.split(v, split_size, dim) for k, v in item.items()}
            # now we transpose it and return List[Dict[key, Tensor]]
            return [
                dict(zip(d.keys(), values))  # type: ignore
                for values in zip(*(d[k] for k in d.keys()))
            ]
        else:
            raise TypeError(f"Cannot split type {type(item)}")

    @staticmethod
    def join(items: List[Chunk], dim=0) -> Chunk:
        """
        Join chunks back together, the inverse of split.

        :param items:
            All the output chunks. Each chunk may be a tensor or a group of
            tensors.
        :param dim:
            The dimension to join along.
        """
        if len(items) == 0:
            raise IndexError("Cannot rejoin an empty list of chunks.")
        item0 = items[0]
        if isinstance(item0, torch.Tensor):
            # base case
            return torch.cat(items, dim=dim)  # type: ignore
        elif isinstance(item0, tuple):
            return tuple(
                PipelineHelper.join(x, dim=dim) for x in zip(*items)
            )  # type: ignore
        elif isinstance(item0, dict):
            keys = item0.keys()
            return {  # type: ignore
                k: PipelineHelper.join([c[k] for c in items], dim=dim)  # type: ignore
                for k in keys
            }
        else:
            raise TypeError(f'Cannot join list of type {type(item0)}')

    @staticmethod
    def chunk_to(chunk: Chunk, device: str) -> Chunk:
        """
        Move the chunk to the device.

        Handles chunks which are groups of tensors.
        """
        if isinstance(chunk, torch.Tensor):
            return chunk.to(device)  # type: ignore
        elif isinstance(chunk, tuple):
            return tuple(
                PipelineHelper.chunk_to(c, device) for c in chunk
            )  # type: ignore
        elif isinstance(chunk, dict):
            return {
                k: PipelineHelper.chunk_to(v, device) for k, v in chunk.items()
            }  # type: ignore
        else:
            raise TypeError('chunk_to only compatible with tensors, tuples or dicts.')

    @staticmethod
    def schedule_work_items(layers: torch.nn.ModuleList, chunks: List[Chunk]):
        """
        Iterate through chunks and layers that should be pipelined.

        Each iteration of this generator yields the following properties:

            - layer_nos: a list of indices of layers for you to forward through
            - chunk_idx: the index of the chunk we are manipulating. Use this
              if you need to update chunk representations.
            - next_device: where the chunk should be moved to AFTER the layer
              computation is done.
        """
        # We want to pipeline our computations so that each GPU is working on
        # chunks of the problem at the same of the time. The load of the will
        # look like this, assuming there are 5 chunks (A, B, C, D, E) and 4
        # GPUs. Each slot fill means that gpu is working on that chunk.
        #
        #         +-----------------+
        #         |       Time      |
        #         | 1 2 3 4 5 6 7 8 |
        # +-------+-----------------+
        # |  G  0 | A B C D E       |
        # |  P  1 |   A B C D E     |
        # |  U  2 |     A B C D E   |
        # |     3 |       A B C D E |
        # +-------+-----------------+
        #
        # Note that some GPUs will be idle much of the time. In reality, we
        # will use 2 * num_gpus as the number of chunks, to minimize idle
        # time.
        num_chunks = len(chunks)
        for l in layers:
            if not hasattr(l, '_mp_gpu'):
                raise RuntimeError(
                    'You must run PipelineHelper.make_parallel on the ModuleList '
                    'before you can use iterate_layers_chunks.'
                )

        # devices maps device_idx -> (device, [layer_idx, layer_idx, ...])
        # for example, if devices is 2 and there are 4 layers, we might have
        #   devices = {
        #     0: ('cuda:0', [0]),
        #     1: ('cuda:1', [1, 2, 3]),
        #   }
        # This means layers 0 is on cuda:0, but layers 1-3 are on cuda:1.
        devices = {
            device_idx: (dev, list(grp))
            for device_idx, (dev, grp) in enumerate(
                itertools.groupby(range(len(layers)), lambda x: layers[x]._mp_gpu)
            )
        }
        num_timesteps = len(devices) + num_chunks
        for timestep in range(num_timesteps):
            for chunk_idx in range(num_chunks):
                device_idx = timestep - chunk_idx
                if device_idx >= 0 and device_idx < len(devices):
                    dev, layers_nos = devices[device_idx]
                    next_device, _ = devices[(device_idx + 1) % len(devices)]
                    assert device_idx in devices
                    yield PipelineWorkItem(
                        chunk_idx=chunk_idx,
                        layer_nos=layers_nos,
                        next_device=next_device,
                    )
