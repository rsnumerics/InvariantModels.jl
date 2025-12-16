# InvariantModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsnumerics.github.io/InvariantModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsnumerics.github.io/InvariantModels.jl/dev/)
[![Build Status](https://github.com/rsnumerics/InvariantModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rsnumerics/InvariantModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/1011222024.svg)](https://doi.org/10.5281/zenodo.17923982)


## A Julia package to calculate data-driven and equation-driven reduced order models

Watch the video explaining invariant foliations [here](https://youtu.be/1SxnRrKN6Bg) and [here](https://www.youtube.com/watch?v=LPGuMT13zMA).

The key concept behind methods implemented here is invariance. The following methods are implemented both for autonomous, forced and parameter dependent systems

  * Invariant foliations from data
  * Two-dimensional invariant manifolds from differential equations and discrete-time systems (maps)
  * Accurate instantaneous frequency and damping ratio calculations

The theory and background on invariant foliations are described in the following papers:

  1. *R. Szalai*, Data-driven modelling of autonomous and forced dynamical systems *[preprint](https://arxiv.org/abs/2512.12432)*, 2025
  2. *R. Szalai*, Non-resonant invariant foliations of quasi-periodically forced systems, *[preprint](https://arxiv.org/abs/2403.14771)*, 2024
  3. *R. Szalai*, Data-Driven Reduced Order Models Using Invariant Foliations, Manifolds and Autoencoders, J Nonlinear Sci 33, 75 (2023). [link](https://doi.org/10.1007/s00332-023-09932-y)
  4. *R. Szalai*, Invariant spectral foliations with applications to model order reduction and synthesis. Nonlinear Dyn 101, 2645–2669 (2020). [link](https://doi.org/10.1007/s11071-020-05891-1)
 
Paper [4] introduced the idea of using invariant foliations for reduced order modelling, paper [3] has shown that only invariant foliations can be used for genuine data-driven reduced order modelling (when we classify all possible methods into: a) autoencoders, b) invariant foliations, c) invariant manifolds, d) equation-free models. Paper [1] makes the method widely applicable and forms the basis of this software.

There are a number of examples

  * [Shaw-Pierre model](Examples/Shaw-Pierre_Oblique)
  * [Car following traffic model](Examples/Car_Follow)
  * [A jointed beam experiment](Examples/Jointed_Beam)
  * [Brake-Reuss beam experiment](Examples/Brake-Reuss_Beam)
  * [Titanium beam experiment](Examples/Gravity_Beam)
  * [A ten-dimensional synthetic example](Examples/Ten-Dimensional)
  * [A building structure](Examples/Building_Model)

This package makes [the previous versions](https://rs1909.github.io/FMA/) [obsolete](https://github.com/rs1909/InvariantModels).
