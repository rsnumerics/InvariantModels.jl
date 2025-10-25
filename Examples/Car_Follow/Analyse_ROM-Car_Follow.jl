Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")
module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels
#
using CairoMakie
using GLMakie

GLMakie.activate!()

function Car_Follow_VF!(dz, z, u, Parameters)
    Amplitude = Parameters.Amplitude
    NDIM = 2 * 5 - 1
    L = NDIM + 1
    v0 = 1
    a = 0.75
    F = Amplitude * u[1]
    dz .= [
        a * (
            -(((-5 + L)^2 * v0) / (50 + (-10 + L) * L)) - z[1] +
            (v0 * (L - 5 * (1 + z[6] + z[7] + z[8] + z[9]))^2) /
            (25 * (1 + (L - 5 * (1 + z[6] + z[7] + z[8] + z[9]))^2 / 25))
        ),
        a * (
            -z[2] +
            25 *
            v0 *
            (
                1 / (50 + (-10 + L) * L) -
                1 / (50 + L^2 + 10 * L * (-1 + z[6]) + 25 * (-2 + z[6]) * z[6])
            )
        ),
        a * (
            -z[3] +
            25 *
            v0 *
            (
                1 / (50 + (-10 + L) * L) -
                1 / (50 + L^2 + 10 * L * (-1 + z[7]) + 25 * (-2 + z[7]) * z[7])
            )
        ),
        a * (
            -z[4] +
            25 *
            v0 *
            (
                1 / (50 + (-10 + L) * L) -
                1 / (50 + L^2 + 10 * L * (-1 + z[8]) + 25 * (-2 + z[8]) * z[8])
            )
        ),
        a * (
            -(((-5 + L)^2 * v0) / (50 + (-10 + L) * L)) - z[5] +
            ((F + v0) * (-1 + L / 5 + z[9])^2) / (1 + (-1 + L / 5 + z[9])^2)
        ),
        z[1] - z[2],
        z[2] - z[3],
        z[3] - z[4],
        z[4] - z[5]
    ]
    return dz
end

function Car_Follow_Forcing!(u, Alpha, Parameters)
    u[1] = dot(Parameters.Weights, Alpha)
    return u
end

function Car_Follow_Forcing_Matrix!(x, Parameters, t)
    return Rigid_Rotation_Matrix!(x, Parameters.Forcing_Grid, Parameters.Omega, t)
end

Name = "Car_Follow"
# VER = "Autonomous"
# DATAVER = "Autonomous"
VER = "Forced"
DATAVER = "Forced"

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


Select = 3
Index = 1
ODE_Select_List = ([1; 2], [1; 2], [1; 2], [1; 2])
MAP_Select_List = ([1; 2], [1; 2], [1; 2], [1; 2])
Model_Radius_List = (0.4, 0.2, 0.4, 0.2)
Data_Radius_List = (0.4, 0.54, 0.2, 0.24)
Implicit_Radius_List = (1.0, 0.54, 0.6, 0.24)

IC_Force = [0.1]

dd = JLSO.load("MTF-$(Name)-$(VER).bson")
MTF = dd[:MTF]
XTF = dd[:XTF]
MTF_Test = dd[:Test_MTF]
XTF_Test = dd[:Test_XTF]
Error_Trace = dd[:Train_Error_Trace]
Test_Trace = dd[:Test_Error_Trace]

# MTF_Test, XTF_Test = Make_Similar(MTF, XTF, length(Index_List_T) - 1)
# MTF = Re_Target(MTF_Old, (Not_Condensed, Condensed_Linear_Fixed))
# MTF = MTF_Old
fig = Create_Plot()

Model_Results = true

if Model_Results
    Radius = Model_Radius_List[Select]
    Radial_Order = 2
    Radial_Intervals = 96
    Polar_Order = 11

    if #=false=# isfile("ODEMODEL-$(Name)-$(VER).bson")
        dd = JLSO.load("ODEMODEL-$(Name)-$(VER).bson")
        MM = dd[:MM]
        MX = dd[:MX]
        MD = dd[:MD]
    else
        MM, MX, MD = InvariantModels.Model_From_Function_Alpha(
            Car_Follow_VF!,
            Car_Follow_Forcing!,
            p -> Rigid_Rotation_Generator(p.Forcing_Grid, p.Omega),
            IC_Force,
            Parameters;
            State_Dimension = State_Dimension,
            Start_Order = 0,
            End_Order = 3,
        )
        JLSO.save(
            "ODEMODEL-$(Name)-$(VER).bson",
            :MM => MM,
            :MX => MX,
            :MD => MD,
        )
    end

    if #=false=# isfile("ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Select]).bson")
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

    if #=false=# isfile("MAPMODEL-$(Name)-$(VER).bson")
        dd = JLSO.load("MAPMODEL-$(Name)-$(VER).bson")
        MM = dd[:MM]
        MX = dd[:MX]
    else
        MM, MX = Model_From_ODE(
            Car_Follow_VF!,
            Car_Follow_Forcing!,
            Car_Follow_Forcing_Matrix!,
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

    if #=false=# isfile("MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Select]).bson")
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
end

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
# SH = ct.MTF[Index][1].SH
# SH = PM.SH
PPM, PPX = Find_DATA_Manifold(
    MTF,
    XTF,
    SH,
    Index;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    Transformation = TR,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)

# TR[:,:,3:end] .= 0 # remove the unresolved dimensions
Radius = Implicit_Radius_List[Select]

Cheb_Order = 2
Cheb_Intervals = 120
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
    Transformation = Data_Decoder ./ reshape(Data_Scale, 1, 1, :),
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
#
# data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
# List_of_Data        = data[:List_of_Data]
# List_of_Phases      = data[:List_of_Phases]
# #
# # E_WW_Latent_SS, E_TT, Valid = Latent_To_Manifold(PPM, PPX, MTF, XTF, Index, MTF_Cache.Parts[Index].Model_Cache.Values, Encoded_Phase; Transformation=Data_Decoder ./ reshape(Data_Scale,1,1,:))
# # E_WW_Latent = E_WW_Latent_SS + Steady_State * Encoded_Phase
# E_WW_Latent = E_WW_Full
# Data_Raw = List_of_Data[1] - Steady_State * List_of_Phases[1]
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
