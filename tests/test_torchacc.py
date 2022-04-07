import importlib
import os


def is_torchx_available():
    return importlib.util.find_spec('torchacc') is not None


if is_torchx_available():
    os.system('../examples/benchmark/run_user_defined_local.sh')
else:
    print('No torxhacc envoirment. Skip torchacc unit tests.')
