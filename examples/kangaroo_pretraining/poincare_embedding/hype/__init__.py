#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import manifolds
from . import energy_function
import argparse

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
}

MODELS = {
    'distance': energy_function.DistanceEnergyFunction,
    'entailment_cones': energy_function.EntailmentConeEnergyFunction,
}


def build_model(opt, N):
    if isinstance(opt, argparse.Namespace):
        opt = vars(opt)
    K = 0.1 if opt['model'] == 'entailment_cones' else None
    manifold = MANIFOLDS[opt['manifold']](K=K)
    return MODELS[opt['model']](
        manifold,
        dim=opt['dim'],
        size=N,
        sparse=opt['sparse'],
        margin=opt['margin']
    )
