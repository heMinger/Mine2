"""Self-Supervised Semantic Scene Completion with Large-Model Prior and Gaussian Refinement.

This package is a lightweight integration layer that connects:
- S4C (data + self-supervision losses)
- MapAnything (geometric initialization)
- gsplat (fast differentiable rasterization)
- DINOv2 (frozen 2D semantics prior)

It is intentionally minimal and meant as a starting point.
"""
