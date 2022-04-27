import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_gpu_env(process_id, cfg):
    """
    Handle single and multi-GPU / multi-node.
    """

    assert cfg.worker_count != 0
    if cfg.worker_count >= 1 and cfg.gpu_count >= 1:
        cfg.n_gpu = cfg.gpu_count * cfg.worker_count
    elif cfg.worker_count == 1 and cfg.gpu_count == -1:
        cfg.n_gpu = torch.cuda.device_count()
    else:
        raise RuntimeError

    if cfg.n_gpu <= 1:
        cfg.local_rank = 0
        cfg.global_rank = 0
        cfg.is_master_node = True
        return

    cfg.train_batch_size = cfg.train_batch_size // cfg.n_gpu

    # Multi-GPU
    cfg.master_addr = os.environ.get("MASTER_ADDR", "")
    if not cfg.master_addr:
        cfg.master_addr = "127.0.0.1"
    tmp_port = os.environ.get("MASTER_PORT", "")
    if tmp_port:
        cfg.master_port = tmp_port
    cfg.master_port = int(cfg.master_port)

    cfg.worker_id = os.environ.get("RANK", "")
    if not cfg.worker_id:
        cfg.worker_id = "0"
    cfg.worker_id = int(cfg.worker_id)

    cfg.world_size = cfg.n_gpu
    cfg.local_rank = process_id
    cfg.global_rank = cfg.worker_id * cfg.world_size // cfg.worker_count + cfg.local_rank
    cfg.is_master_node = (cfg.global_rank == 0)
    torch.cuda.set_device(cfg.local_rank)
    torch.distributed.init_process_group(
        init_method="tcp://{}:{}".format(cfg.master_addr, cfg.master_port),
        backend="nccl",
        world_size=cfg.world_size,
        rank=cfg.global_rank
    )


def distributed_call_main(main_fn, cfg, *args, **kwargs):
    if cfg.gpu_count == -1:
        num_process = torch.cuda.device_count()
    else:
        num_process = cfg.gpu_count
    if (num_process <= 1 and cfg.worker_count == 1):
        main_fn(0, cfg, *args, **kwargs)
    else:
        mp.spawn(main_fn, nprocs=num_process, args=(cfg, args, kwargs, ))