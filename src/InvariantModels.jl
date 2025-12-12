# SPDX-License-Identifier: EUPL-1.2

module InvariantModels

macro exportinstances(enum)
    eval = GlobalRef(Core, :eval)
    return :($eval($__module__, Expr(:export, map(Symbol, instances($enum))...)))
end

using LinearAlgebra
using Random
using ComponentArrays
using Combinatorics
#
using Printf
using Tullio
#
using RecursiveArrayTools
using Manifolds
using ManifoldsBase
using ManifoldDiff
using Manopt

BLAS.set_num_threads(1)
Tullio._THREADS[] = false

# some utilities
include("Component_Utils.jl")
include("Hessian_Utils.jl")
include("Fourier_Utils.jl")
export Fourier_Grid
export Fourier_Interpolate
export Rigid_Rotation_Matrix!
export Rigid_Rotation_Generator
include("Polynomial_Interpolation.jl")
export Chebyshev_Grid
export Barycentric_Interpolation_Matrix
# vector bundle decomposition
include("Linear_Decomposition.jl")
using StaticArrays
include("Bundle_Decomposition.jl")
export Bundle_Decomposition
export Reduce_Bundles_Lite
# data creation and processing
include("Data_Wrangling.jl")
export Cut_Signals
export Delay_Embed_Proper
export Delay_Embed
export Chop_Signals
export Chop_And_Stitch
export Pick_Signals
export Linear_Decompose
# ---new---
export Estimate_Linear_Model
export Filter_Linear_Model
export Create_Linear_Decomposition
export Select_Bundles_By_Energy
export Decompose_Data
export Decomposed_Data_Scaling

using DifferentialEquations
using Interpolations
using UnicodePlots
include("Generate_From_ODE.jl")
export Generate_Data_From_ODE
# - NEW
export Generate_From_ODE

# quasi-periodic part
include("SkewFunction.jl")
# linear part
include("ArrayStiefel.jl")
include("ArrayStiefelOblique.jl")
include("Mean_Stiefel.jl")
# nonlinear part
include("Tensor.jl")
include("Dense_Polynomial.jl")
# the encoder
include("Encoder.jl")
@exportinstances Encoder_Linear_Type
@exportinstances Encoder_Nonlinear_Type
# the low-dimensional model
using TaylorSeries
import ForwardDiff
include("MultiStep_Model.jl")
export MultiStep_Model
export Unitary_Model
export From_Data!
export Evaluate
export Slice
export Model_From_Function
export Model_From_ODE
# - NEW
export Model_From_Function_Alpha
#
# include("DMD_Model.jl")
# single foliation
include("Foliation.jl")
# set of foliations
using JLSO
include("Multi_Foliation.jl")
@exportinstances Scaling_Type
# export Multi_Foliation
export Make_Similar
# export Make_Cache
# export Optimise_New!
include("Multi_Foliation_Problem.jl")
export Multi_Foliation_Problem
export Multi_Foliation_Test_Problem
export Optimise!
#
using StatsBase
using NonlinearSolve, NLsolve
using LaTeXStrings
using Makie
include("Polar_Manifold.jl")
export Find_DATA_Manifold
export Find_MAP_Manifold
export Create_Plot
export Data_Result
export Data_Error
export Model_Result
export Plot_Backbone_Curves!
export Plot_Error_Curves!
export Plot_Error_Trace
export Annotate_Plot!
include("Polar_Implicit_Manifold.jl")
export Extract_Manifold_Embedding
include("Polar_ODE_Manifold.jl")
export Find_ODE_Manifold
# export Model_Result

end
