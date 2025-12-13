```@meta
CurrentModule = InvariantModels
```

# Invariant Models

Documentation for [InvariantModels](https://github.com/rsnumerics/InvariantModels.jl).

The primary goal of this software package is to identify mathematical models from data. 
The identified model can be forced, parameter dependent or autonomous.
The mathematical structure of the identified model is a series of invariant foliations.
To avoid overfitting the foliation representation can be adjusted. 
In general, encoders are represented as compressed polynomials, while the underlying low-dimensional model is an ordinary multivariate polynomial.

The software calculates invariant manifolds in vector fields and maps using a direct numerical method.
The invariance equation discretised using a mixed Fourier and piecewise-Chebyshev collocation scheme. 
Then the discretised equations are solved using a similar method to pseude-archlength continuation, which employs a phase and arclength condition.
This is much more accurate than using series expansions about the steady state and does not suffer from small radii of convergence.

There are extensive utilities to prepare data for model identification. 
These include delay embedding and dynamic mode decomposition to find ideal delay embedding coordinates.
Approximate linear models can be decomposed into invariant vector bundles to provide an optimal coordinate system for model identification.

## Set-up

The system to be identified is in the skew-product form
```math
\begin{aligned}
    \boldsymbol{x}_{k+1} &= \boldsymbol{f} \left(\boldsymbol{x}_{k}, \boldsymbol{\theta} \right)\\
    \boldsymbol{\theta}_{k+1} &= \boldsymbol{g}( \boldsymbol{\theta}_{k} )
\end{aligned} \tag{MAP},
```
where ``\boldsymbol{f}\in X\times Y\to X``, ``\boldsymbol{g}:Y\to Y`` are functions defined on the ``d_{X}``-dimensional linear space ``X`` and ``d_{Y}``-dimensional differentiable manifold ``Y``, respectively.
The forcing map ``\boldsymbol{g}`` is volume preserving, hence it has recurrent dynamics.

Every volume preserving map can be transformed into a coordinate system where its representation is approximately a unitary matrix. 
To find the unitary matrix ``\boldsymbol{\Omega}`` we use the transformation
```math
\boldsymbol{\alpha} = \boldsymbol{\Psi}(\boldsymbol{\theta}),
```
where ``\boldsymbol{\Psi}`` is a vector of linearly independent scalar valued functions.
It is known that as the dimensionality of ``\boldsymbol{\Psi}`` increases, the volume preserving map ``\boldsymbol{g}`` transforms into a linear unitary map ``\boldsymbol{\Omega}`` [KlusKoopma2016](@cite), [Korda_2017](@cite). Another consequence of the transformation is that, when transformed, function ``\boldsymbol{f}`` becomes linear in new variables ``\boldsymbol{\alpha}``.

Accordingly, system ([MAP]()) is written in the form of
```math
\begin{aligned}
    \boldsymbol{x}_{k+1} &= \boldsymbol{F} \left(\boldsymbol{x}_{k}\right) \boldsymbol{\alpha}\\
    \boldsymbol{\alpha}_{k+1} &= \boldsymbol{\Omega} \boldsymbol{\alpha}_{k},
\end{aligned} \tag{MAP-TR}
```
where ``\boldsymbol{x}_{k} \in \mathbb{R}^n`` is the state variable, ``\boldsymbol{\alpha} \in \mathbb{R}^m`` is the **encoded** phase variable.

## Data representation

The data is a set of ``N`` trajectories, each of which are ``\ell_{j}-\ell_{j-1}`` points long for ``j=1,\ldots,N``, where ``\ell_{0}=0``. In formulae the trajectories are 
```math
\begin{align*}
    \left(\boldsymbol{x}_{1},\boldsymbol{\alpha}_{1}\right),\left(\boldsymbol{x}_{2},\boldsymbol{\alpha}_{2}\right),\ldots,\left(\boldsymbol{x}_{\ell_{1}},\boldsymbol{\alpha}_{\ell_{1}}\right) & \\
    \vdots \qquad \qquad & \\
    \left(\boldsymbol{x}_{\ell_{N-1}+1},\boldsymbol{\alpha}_{\ell_{N-1}+1}\right),\left(\boldsymbol{x}_{\ell_{N-1}+2},\boldsymbol{\alpha}_{\ell_{N-1}+2}\right),\ldots,\left(\boldsymbol{x}_{\ell_{N}},\boldsymbol{\alpha}_{\ell_{N}}\right) &
\end{align*}
```
Here ``\boldsymbol{\alpha}`` represent the state of the forcing dynamics as encoded by the ``\boldsymbol{\Psi}`` function. 

The data is arranged into two arrays. The array `Data` is simply the state
```math
\begin{pmatrix} 
    \boldsymbol{x}_{1} & \boldsymbol{x}_{2} & \cdots & \boldsymbol{x}_{\ell_N}
\end{pmatrix}
```
The second array `Encoded_Phase` contains the forcing states
```math
\begin{pmatrix} 
    \boldsymbol{\alpha}_{1} & \boldsymbol{\alpha}_{2} & \cdots & \boldsymbol{\alpha}_{\ell_N}
\end{pmatrix}
```
To make sure that the system know where each trajectory starts and ends, we also must supply the `Index_List` vector
```math
\begin{pmatrix} 
    \ell_{0} & \ell_{1} & \cdots & \ell_N
\end{pmatrix}.
```

## Why invariant foliations

Invariant foliations are the only architecture that guarantees invariance and uniqueness (under some smoothness and non-resonance conditions) at the same time [Szalai2020, Szalai2023](@cite).

All dynamic model reduction architectures can be put into four categories. 
Each architecture must contain functional connections between the model and the data. 
This functional connection can only go two ways: from the data to the model or in reverse, which are the encoders and decoders, respectively. 
One has to establish this connection both at the initial condition and at the model prediction. This is exactly four combinations.

There are four ways to connect a low-order model ``\boldsymbol{R}`` to ``\boldsymbol{F}``. The figure below shows the four combinations. Only invariant foliations and invariant manifolds produce meaningful reduced order models. Only invariant foliations and autoencoders can be fitted to data. The intersection is invariant foliations.

![](FourDiag-Plain.svg)

Therefore,
* when a **system of equations** is given, invariant manifolds are the most appropriate (foliations are still possible),
* when **data** is given, only invariant foliations are appropriate.

Autoencoders such as [SSMLearn](https://github.com/haller-group/SSMLearn) do not enforce invariance and therefore generate spurious results as shown in [Szalai2023](@cite).

## Invariant Foliations

The sofware identifies the encoder ``\boldsymbol{U}`` and the conjugate map ``\boldsymbol{R}`` from the invariance equation
```math
\boldsymbol{R}\left(\boldsymbol{U}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}\right) = \boldsymbol{U}\left(\boldsymbol{f}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{g}\left(\boldsymbol{\theta}\right)\right). \tag{FOIL}
```
Equation ([FOIL]()) is turned into a least squares optimisation problem and the loss function is minimised.

## Invariant Manifolds

The sofware also calculates invariant manifolds from maps and vector fields. The invariance for manifolds is
```math
\boldsymbol{S}\left(\boldsymbol{V}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}\right) = \boldsymbol{V}\left(\boldsymbol{f}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{g}\left(\boldsymbol{\theta}\right)\right), \tag{MAN}
```
where ``\boldsymbol{V}`` is the decoder and ``\boldsymbol{S}`` is the conjugate map.
Since we are interested in oscilllatory dynamics on 2D manifolds, the invariance equation can be written in polar coordinates, that is
```math
\boldsymbol{V}\left(R\left(\rho\right),T\left(\rho\right),\boldsymbol{g}\left(\boldsymbol{\theta}\right)\right)=\boldsymbol{f}\left(\boldsymbol{V}\left(\rho,\beta,\boldsymbol{\theta}\right),\boldsymbol{\theta}\right),
```
where ``R`` and ``T`` represent the conjugate dynamics. The parametrisation of the invariant manifold is fixed by the amplitude and phase conditions
```math
\begin{aligned}
    \int_{Y}\int_{0}^{2\pi}\left\langle D_{1}\boldsymbol{V}\left(\rho,\beta,\boldsymbol{\theta}\right),\boldsymbol{V}\left(\rho,\beta,\boldsymbol{\theta}\right)-\boldsymbol{V}\left(0,\beta,\boldsymbol{\theta}\right)\right\rangle \mathrm{d}\beta\mathrm{d}\boldsymbol{\theta} &= \rho \\
    \int_{Y}\int_{0}^{2\pi}\left\langle D_{1}\boldsymbol{V}\left(r,\beta,\boldsymbol{\theta}\right),D_{2}\boldsymbol{V}\left(r,\beta,\boldsymbol{\theta}\right)\right\rangle \mathrm{d}\beta\mathrm{d}\boldsymbol{\theta} &= 0.
\end{aligned}
```
The instantantaneous damping ratio and frequency are calculated as
```math
\begin{aligned}
\zeta\left(\rho\right)	&=-\frac{\frac{\mathrm{d}}{\mathrm{d}\rho}R\left(\rho\right)}{T\left(\rho\right)}\;\;\text{and}\\
\omega\left(\rho\right)	&=T\left(\rho\right),
\end{aligned}
```
respectively.

## References

```@bibliography
Pages = ["index.md"]
Canonical=false
```

## Index

```@index
```

## API

### Fourier collocation 
```@docs
Fourier_Grid
```

```@docs
Fourier_Interpolate
```

```@docs
Rigid_Rotation_Matrix!
```

```@docs
Rigid_Rotation_Generator
```

### One-dimensional polynomial collocation and interpolation

```@docs
Chebyshev_Grid
```

```@docs
Barycentric_Interpolation_Matrix
```

### Data pre-processing 

```@docs
Delay_Embed
```

```@docs
Chop_And_Stitch
```

```@docs
Estimate_Linear_Model
```

```@docs
Filter_Linear_Model
```

```@docs
Create_Linear_Decomposition
```

```@docs
Select_Bundles_By_Energy
```

```@docs
Decompose_Data
```

```@docs
Decomposed_Data_Scaling
```

### Generating data from differential equations

```@docs
Generate_From_ODE
```

### Types of encoders

```@docs
Encoder_Linear_Type
```

```@docs
Encoder_Nonlinear_Type
```

### Multivariate polynomial models

These models represent
* The conjugate dynamics
* a vector field
* a nonlinear map produced from a vector field

The representation is used to fit full trajectories to data and therefore making it a highly nonlinear optimisation problem.

```@docs
MultiStep_Model
```

```@docs
From_Data!
```

```@docs
Evaluate
```

```@docs
Slice
```

```@docs
Model_From_ODE
```

```@docs
Model_From_Function_Alpha
```

### Representing invariant foliations

```@docs
Scaling_Type
```

```@docs
Make_Similar
```

```@docs
Multi_Foliation_Problem
```

```@docs
Multi_Foliation_Test_Problem
```

```@docs
Optimise!
```

### Analysing invariant foliations

```@docs
Find_DATA_Manifold
```

```@docs
Extract_Manifold_Embedding
```

```@docs
Data_Result
```

```@docs
Data_Error
```

### Calculating invariant manifolds from ODEs and maps

```@docs
Find_ODE_Manifold
```

```@docs
Find_MAP_Manifold
```

```@docs
Model_Result
```

### Plotting results

```@docs
Create_Plot
```

```@docs
Plot_Error_Curves!
```

```@docs
Plot_Error_Trace
```

```@docs
Annotate_Plot!
```
