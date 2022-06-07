import torch
import logging
from mpi4py import MPI
import numpy as np


def proc_id():
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    return MPI.COMM_WORLD.Get_size()


def allreduce(*args, **kwargs):
    try:
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)
    except Exception as e:
        logging.error("x={}".format(args[0]))
        logging.error("buff={}".format(args[1]))
        logging.error(e)
        raise


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_avg(x):
    return mpi_op(x, MPI.SUM) / num_procs()


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_avg_grad(module, device):
    """MPIプロセス間の勾配の平均化"""
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_grad_numpy = p.grad.cpu().numpy()
        avg_p_grad = mpi_avg(p_grad_numpy)
        p_grad_numpy[:] = avg_p_grad[:]
        p.grad = torch.as_tensor(
            p_grad_numpy,
            dtype=torch.float32,
            device=device
        )


def sync_params(module, device):
    """パラメータをMPIプロセス間で同期"""
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_numpy = p.data.cpu().numpy()
        broadcast(p_numpy)
        p.data = torch.as_tensor(
            p_numpy,
            dtype=torch.float32,
            device=device
        )
