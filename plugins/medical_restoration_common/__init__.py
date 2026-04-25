"""Shared helpers for medical image restoration plugins.

Backends (CoreDiff, SNRAware, Fast-DDPM) live outside the main repo under
``external_backends/`` and have their own runtimes. This package lazy-loads
adapters so the ChatClinic backend can still start when weights are missing.
"""
