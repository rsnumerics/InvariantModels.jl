Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")
module ct
using LinearAlgebra
using Random
using ComponentArrays
using Combinatorics
using Manifolds
using ManifoldsBase
using ManifoldDiff
using Tullio
using RecursiveArrayTools
using Manopt
using TaylorSeries
using Printf
using Interpolations
using DifferentialEquations
using UnicodePlots
using CairoMakie
using GLMakie
# using Wavelets, WaveletsExt
using DSP
using LoopVectorization
using FastChebInterp
using StaticArrays
using StatsBase
using NonlinearSolve
import DelaunayTriangulation
import NaturalNeighbours
import ForwardDiff
using ApproxFun

GLMakie.activate!()

# using Clustering
using JLSO
using MAT
BLAS.set_num_threads(1)
using Main.InvariantModels

# t1 is within period, t2 quasiperiodic
function Shaw_Pierre_VF!(x, y, u, Parameters)
    # u[1] = cos(Omega_ODE * t)
    # u[2] = cos(Omega_ODE * t + 0.1)
    # u[3] = Kappa -> 0.5 -> Nonlinearity
    Amplitude = Parameters.Amplitude
    k1 = 1.0
    k2 = 3.325
    Kappa = u[3]
    c1 = 0.05
    c2 = 0.01
    #     c1 = 0.01
    #     c2 = 0.002
    x[1] = y[3]
    x[2] = y[4]
    x[3] =
        -k1 * y[1] + k2 * y[2] - c1 * y[3] + c2 * y[4] - Kappa * (y[1]^3) +
        Kappa * c1 * (y[3]^3) +
        Amplitude * u[2]
    x[4] = -(k1 + 2 * k2) * y[2] - (c1 + 2 * c2) * y[4] + Amplitude * u[1]
    return x
end

function Shaw_Pierre_Forcing!(u, Alpha, Parameters)
    u[1:2] .= 0
    u[3] = 0.5
    return u
end

function Shaw_Pierre_Forcing_Matrix!(x, Parameters, t)
    return I
end

Name = "SP_Oblique"
VER = "Parametric"
DATAVER = "Parametric"

data = JLSO.load("DATA-$(Name)-$(DATAVER).bson")
Parameters      = data[:Parameters]
Time_Step       = data[:Time_Step]
Index_List      = data[:Index_List]
Data_Decomp     = data[:Data_Decomp]
Encoded_Phase   = data[:Encoded_Phase]
Index_List_T    = data[:Index_List_T]
Data_Decomp_T   = data[:Data_Decomp_T]
Encoded_Phase_T = data[:Encoded_Phase_T]
Decomp          = data[:Decomp]
Steady_State    = data[:Steady_State]
Linear_Model    = data[:Linear_Model]
SH              = data[:SH]
Data_Scale      = data[:Data_Scale]
#
Data_Encoder      = Decomp.Data_Encoder
Data_Decoder      = Decomp.Data_Decoder
Omega_ODE         = Parameters.Omega
Forcing_Amplitude = Parameters.Amplitude
State_Dimension   = size(Data_Decomp, 1)
Skew_Dimension    = size(Encoded_Phase, 1)

Select = 4
Index = 1
Model_Radius_List = (1.0, 0.5, 1.0, 1.0)
Data_Radius_List = (1.0, 0.5, 1.0, 1.0) # for I=2 -> 0.14
Implicit_Radius_List = (1.0, 0.5, 1.0, 1.0)
ODE_Select_List = ([1; 2], [1; 2], [1; 2], [1; 2])
MAP_Select_List = ([1; 2], [1; 2], [1; 2], [1; 2])

IC_Force = [0.0; 0.0; 0.5]

dd = JLSO.load("MTF-$(Name)-$(VER).bson")
MTF = dd[:MTF]
XTF = dd[:XTF]
MTF_Test = dd[:Test_MTF]
XTF_Test = dd[:Test_XTF]
Error_Trace = dd[:Train_Error_Trace]
Test_Trace = dd[:Test_Error_Trace]

# Encoded_Slice = vec(Barycentric_Interpolation_Matrix(Forcing_Grid, [0.5]))
Encoded_Slice = Encoded_Phase_T[:, rand(axes(Encoded_Phase_T, 2))]
MTF_Slice, XTF_Slice = Slice(MTF, XTF, Encoded_Slice)

fig = Create_Plot()

Radius = Model_Radius_List[Select]
Radial_Order = 2
Radial_Intervals = 96
Polar_Order = 11

if false # isfile("ODEMODEL-$(Name)-$(VER).bson")
    dd = JLSO.load("ODEMODEL-$(Name)-$(VER).bson")
    MM = dd[:MM]
    MX = dd[:MX]
    MD = dd[:MD]
else
    MM, MX, MD = InvariantModels.Model_From_Function_Alpha(
        Shaw_Pierre_VF!,
        Shaw_Pierre_Forcing!,
        p -> Rigid_Rotation_Generator(p.Forcing_Grid, p.Omega),
        IC_Force,
        Parameters;
        State_Dimension = State_Dimension,
        Start_Order = 0,
        End_Order = 3,
    )
    MM, MX = Slice(MM, MX, Encoded_Slice)
    MD = zeros(1, 1)
    JLSO.save(
        "ODEMODEL-$(Name)-$(VER).bson",
        :MM => MM,
        :MX => MX,
        :MD => MD,
    )
end

if false # isfile("ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Select]).bson")
    dd = JLSO.load(
        "ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Select]).bson",
    )
    MP = dd[:MP]
    XP = dd[:XP]
else
    MP, XP = Find_ODE_Manifold(
        MM,
        MX,
        MD,
        ODE_Select_List[Select];
        Radial_Order = Radial_Order,
        Radial_Intervals = Radial_Intervals,
        Radius = Radius,
        Phase_Dimension = Polar_Order,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 32,
        initial_maxiters = 200,
    )
    JLSO.save(
        "ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Select]).bson",
        :MP => MP,
        :XP => XP,
    )
end

Plot_Model_Result!(fig, MP, XP, Hz = false)

if false # isfile("MAPMODEL-$(Name)-$(VER).bson")
    dd = JLSO.load("MAPMODEL-$(Name)-$(VER).bson")
    MM = dd[:MM]
    MX = dd[:MX]
else
    MM, MX = Model_From_ODE(
        Shaw_Pierre_VF!,
        Shaw_Pierre_Forcing!,
        Shaw_Pierre_Forcing_Matrix!,
        IC_Force,
        Parameters,
        Time_Step / 512,
        Time_Step,
        State_Dimension = State_Dimension,
        Skew_Dimension = Skew_Dimension,
        Start_Order = 0,
        End_Order = 5,
        Steady_State = true,
    )
    MM, MX = Slice(MM, MX, Encoded_Slice)
    JLSO.save(
        "MAPMODEL-$(Name)-$(VER).bson",
        :MM => MM,
        :MX => MX,
    )
end

Radius = Model_Radius_List[Select]
Cheb_Order = 2
Cheb_Intervals = 96
Polar_Order = 11

if false #=isfile("MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Select]).bson")=#
    dd = JLSO.load(
        "MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Select]).bson",
    )
    PM = dd[:PM]
    PX = dd[:PX]
else
    PM, PX = Find_MAP_Manifold(
        MM,
        MX,
        MAP_Select_List[Select];
        Radial_Order = Cheb_Order,
        Radial_Intervals = Cheb_Intervals,
        Radius,
        Phase_Dimension = Polar_Order,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 32,
        initial_maxiters = 200,
    )
    JLSO.save(
        "MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Select]).bson",
        :PM => PM,
        :PX => PX,
    )
end

Plot_Model_Result!(
    fig,
    PM,
    PX,
    Time_Step = Time_Step,
    Hz = false,
    Damping_By_Derivative = true,
)
display(fig)

Radius = Data_Radius_List[Select]
# if Select < 3
Cheb_Order = 2
Cheb_Intervals = 112
# else
#     Cheb_Order = 5
#     Cheb_Intervals = 24
# end
Polar_Order = 17

# shift required from previous model
TR = Data_Decoder ./ reshape(Data_Scale, 1, 1, :)
@tullio TR_Slice_R[i, k] := TR[i, j, k] * Encoded_Slice[j]
display(TR_Slice_R)
TR_Slice = reshape(TR_Slice_R, size(TR_Slice_R, 1), 1, size(TR_Slice_R, 2))
SH = ones(1,1)
PPM, PPX = Find_DATA_Manifold(
    MTF_Slice,
    XTF_Slice,
    SH,
    Index;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    Transformation = TR_Slice,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)

# TR[:,:,3:end] .= 0 # remove the unresolved dimensions
Radius = Implicit_Radius_List[Select]

Cheb_Order = 2
Cheb_Intervals = 240
Polar_Order = 17
MIP, XIP, Torus, E_WW_Full, Latent_Data, E_ENC, AA, Valid_Ind = Extract_Manifold_Embedding(
    MTF,
    XTF,
    Index,
    Data_Decomp,
    Encoded_Phase;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    Output_Transformation = Data_Encoder,
    Output_Scale = vec(Data_Scale),
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)

# Latent_Data = To_Latent(MTF, XTF, Index, Data_Decomp, Encoded_Phase)
# E_WW, E_TT, Valid = Latent_To_Manifold(PPM, PPX, MTF, XTF, Index, Latent_Data, Encoded_Phase; Transformation=Data_Decoder ./ reshape(Data_Scale,1,1,:))

# fig = Create_Plot()
MTF_Cache = Plot_Data_Result!(
    fig,
    PPM,
    PPX,
    MIP,
    XIP,
    MTF,
    XTF,
    Index,
    Index_List,
    Data_Decomp,
    Encoded_Phase,
    Transformation = TR,
    Slice_Transformation = TR_Slice,
    Time_Step = Time_Step,
    Hz = false,
    Damping_By_Derivative = true,
)
MTF_Cache, Data_Max = Plot_Data_Error!(
    fig,
    PPM,
    PPX,
    MIP,
    XIP,
    MTF_Test,
    XTF_Test,
    Index,
    Index_List_T,
    Data_Decomp_T,
    Encoded_Phase_T;
    Transformation = Data_Decoder ./ reshape(Data_Scale, 1, 1, :),
    Color = Makie.wong_colors()[2],
    Model_IC = true,
)
Plot_Error_Trace(fig, Index, Error_Trace, Test_Trace)


Annotate_Plot!(fig)
# axFreq = content(fig[1, 4])
# axDamp = content(fig[1, 5])
# xlims!(axFreq, 1.0, 1.03)
# xlims!(axDamp, 0.0245, 0.0255)
# ylims!(axFreq, 0, 0.5)
# ylims!(axDamp, 0, 0.5)


CairoMakie.activate!(type = "svg")
save("FIGURE-$(Name)-$(VER)-I$(Index).svg", fig)
GLMakie.activate!()

fig2 = Figure()
# Latent data
Latent_Dimension = size(MTF_Cache.Parts[Index].Latent_Data, 1)
Rows = round(Int, sqrt(Latent_Dimension))
X_Axis = range(0, step = Time_Step, length = size(Data_Decomp_T, 2))
let it = 0
    let k = Index
        for l = 1:size(MTF_Cache.Parts[k].Latent_Data, 1)
            id = it + 1
            ax = Makie.Axis(
                fig2[1+div(it, Rows), 1+mod(it, Rows)],
                xlabel = L"$t$ [s]",
                ylabel = L"z_{%$id}",
            )
            lines!(
                ax,
                X_Axis,
                MTF_Cache.Parts[k].Latent_Data[l, :],
                color = Makie.wong_colors()[1],
            )
            lines!(
                ax,
                X_Axis,
                MTF_Cache.Parts[k].Model_Cache.Values[l, :],
                color = Makie.wong_colors()[2],
            )
            lines!(
                ax,
                X_Axis,
                MTF_Cache.Parts[k].Latent_Data[l, :] -
                MTF_Cache.Parts[k].Model_Cache.Values[l, :],
                color = Makie.wong_colors()[3],
            )
            it += 1
        end
    end
end
# Model data
#     let it = 0
#         let k = Index
#             for l in 1:size(MTF_Cache.Parts[k].Latent_Data, 1)
#                 ax = content(fig2[1+div(it,Rows),1+mod(it,Rows)])
#                 lines!(ax, X_Axis, MTF_Cache.Parts[k].Model_Cache.Values[l,:], color=Makie.wong_colors()[2])
#                 it += 1
#             end
#         end
#     end
display(fig2)
CairoMakie.activate!(type = "svg")
save(
    "FIGURE-LATENT-$(Name)-$(VER)-I$(Index).svg",
    fig2,
)
GLMakie.activate!()

# fig3 = Figure()
# #
# # E_WW_Latent_SS, E_TT, Valid = Latent_To_Manifold(PPM, PPX, MTF, XTF, Index, MTF_Cache.Parts[Index].Model_Cache.Values, Encoded_Phase; Transformation=Data_Decoder ./ reshape(Data_Scale,1,1,:))
# # E_WW_Latent = E_WW_Latent_SS + SS * Encoded_Phase
# E_WW_Latent = E_WW_Full
# Data_Raw = ct.data[:Raw_Data] - SS * Encoded_Phase
# X_Axis = range(0, step = ct.Time_Step, length = size(Data_Raw, 2))
# #
# Invalid = findall(.!Valid_Ind)
# #
# Rows = round(Int, sqrt(State_Dimension))
# let it = 0
#     for k = 1:size(E_WW_Latent, 1)
#         ax = Makie.Axis(
#             fig3[1+div(it, Rows), 1+mod(it, Rows)],
#             xlabel = L"$t$ [s]",
#             ylabel = L"z_{%$k}",
#         )
#         lines!(ax, X_Axis, Data_Raw[k, :], color = Makie.wong_colors()[1])
#         it += 1
#     end
# end
# #
# let it = 0
#     for k = 1:size(E_WW_Latent, 1)
#         ax = content(fig3[1+div(it, Rows), 1+mod(it, Rows)])
#         lines!(ax, X_Axis, E_WW_Latent[k, :], color = Makie.wong_colors()[2])
#         scatter!(
#             ax,
#             X_Axis[Invalid],
#             E_WW_Latent[k, Invalid],
#             color = Makie.wong_colors()[3],
#         )
#         it += 1
#     end
# end
# CairoMakie.activate!(type = "svg")
# save(
#     "FIGURE-MANIF-$(Name)-$(VER)-I$(Index).svg",
#     fig3,
# )
# GLMakie.activate!()
end
