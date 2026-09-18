"""Analytic tight-binding toy models (permanent regression fixtures)."""

from __future__ import annotations

from wannierx.models.chain import chain
from wannierx.models.graphene import graphene
from wannierx.models.haldane import haldane
from wannierx.models.qzhang import qiwuzhang
from wannierx.models.ssh import ssh

__all__ = ["chain", "graphene", "haldane", "qiwuzhang", "ssh"]
