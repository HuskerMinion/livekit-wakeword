"""Tests for shared utilities."""

from __future__ import annotations

import pytest
import torch

from livekit.wakeword.utils import get_device


@pytest.fixture
def no_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_cpu_fallback(no_accelerators):
    assert get_device() == torch.device("cpu")


def test_cuda_first(no_accelerators, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert get_device() == torch.device("cuda")


@pytest.mark.skipif(not hasattr(torch, "xpu"), reason="torch build has no torch.xpu")
def test_xpu_before_mps(no_accelerators, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device() == torch.device("xpu")
