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
    u[1:2] .= Parameters.Weights * Alpha
    u[3] = 0.5
    return u
end

function Shaw_Pierre_Forcing_Matrix!(x, Parameters, t)
    return Rigid_Rotation_Matrix!(x, Parameters.Forcing_Grid, Parameters.Omega, t)
end

Name = "SP_Oblique"
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
Steady_State    = data[:Decomp].Steady_State # data[:Steady_State]
# Linear_Model    = data[:Linear_Model]
SH              = data[:Decomp].SH # data[:SH]
Data_Scale      = data[:Data_Scale]
#
Data_Encoder      = Decomp.Data_Encoder
Data_Decoder      = Decomp.Data_Decoder
Omega_ODE         = Parameters.Omega
Forcing_Amplitude = Parameters.Amplitude
State_Dimension   = size(Data_Decomp, 1)
Skew_Dimension    = size(Encoded_Phase, 1)

Model_Radius_List = (1.0, 1.0,)
Data_Radius_List = (1.0, 1.0,) # for I=2 -> 0.14
Implicit_Radius_List = (1.2, 1.2,)
ODE_Select_List = ([1; 2], [3; 4],)
MAP_Select_List = ([1; 2], [3; 4],)

IC_Force = [0.0; 0.0; 0.5]

# MTFP = Multi_Foliation_Problem(
#     Index_List,
#     Data_Decomp,
#     Encoded_Phase,
#     Selection = ([1; 2], [3; 4]),
#     Model_Orders = (3, 1),
#     Encoder_Orders = (3, 1),
#     Unreduced_Model = Decomp.Unreduced_Model,
#     Reduced_Model = Decomp.Reduced_Model,
#     Reduced_Encoder = Decomp.Reduced_Encoder,
#     SH = SH,
#     Initial_Iterations = 32,
#     Scaling_Parameter = 2^(-2),
#     Initial_Scaling_Parameter = 2^(-2),
#     Scaling_Order = Linear_Scaling,
#     node_ratio = 0.8,
#     leaf_ratio = 1.0,
#     max_rank = 24,
#     Linear_Type = (Encoder_Array_Stiefel, Encoder_Mean_Stiefel),
#     Nonlinear_Type = (Encoder_Dense_Latent_Linear, Encoder_Dense_Full),
# #     Linear_Type = (Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
# #     Nonlinear_Type = (Encoder_Dense_Full, Encoder_Dense_Full),
#     Name = "MTF-$(Name)-$(VER)",
#     Time_Step = Time_Step,
#     Train_Model = false,
# )
# MTFP_Test = Multi_Foliation_Test_Problem(
#     MTFP,
#     Index_List_T,
#     Data_Decomp_T,
#     Encoded_Phase_T;
#     Initial_Scaling_Parameter = 2^(-2),
# )

dd = JLSO.load("MTF-$(Name)-$(VER).bson")
# dd[:MTF] = MTFP.MTF
# dd[:Test_MTF] = MTFP_Test.MTF
# JLSO.save("MTF-$(Name)-$(VER).bson", dd)

MTF = dd[:MTF]
XTF = dd[:XTF]
MTF_Test = dd[:Test_MTF]
XTF_Test = dd[:Test_XTF]
Error_Trace = dd[:Train_Error_Trace]
Test_Trace = dd[:Test_Error_Trace]

Model_Results = true

fig1_List = []
fig2_List = []
fig3_List = []

for Index in 1:2

if Model_Results
    Radius = Model_Radius_List[Index]
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
            Shaw_Pierre_VF!,
            Shaw_Pierre_Forcing!,
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

    if #=false=# isfile("ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Index]).bson")
        dd = JLSO.load(
            "ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Index]).bson",
        )
        MP = dd[:MP]
        XP = dd[:XP]
    else
        MP, XP = Find_ODE_Manifold(
            MM,
            MX,
            MD,
            ODE_Select_List[Index];
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
            "ODEMANIFOLD-$(Name)-$(VER)-S$(ODE_Select_List[Index]).bson",
            :MP => MP,
            :XP => XP,
        )
    end

    ODE_Backbone = Model_Result(MP, XP, Hz = false)

    if #=false=# isfile("MAPMODEL-$(Name)-$(VER).bson")
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
        JLSO.save(
            "MAPMODEL-$(Name)-$(VER).bson",
            :MM => MM,
            :MX => MX,
        )
    end

    Radius = Model_Radius_List[Index]
    Cheb_Order = 2
    Cheb_Intervals = 96
    Polar_Order = 11

    if #=false=# isfile("MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Index]).bson")
        dd = JLSO.load(
            "MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Index]).bson",
        )
        PM = dd[:PM]
        PX = dd[:PX]
    else
        PM, PX = Find_MAP_Manifold(
            MM,
            MX,
            MAP_Select_List[Index];
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
            "MAPMANIFOLD-$(Name)-$(VER)-S$(MAP_Select_List[Index]).bson",
            :PM => PM,
            :PX => PX,
        )
    end

    MAP_Backbone = Model_Result(
        PM,
        PX,
        Time_Step = Time_Step,
        Hz = false,
        Damping_By_Derivative = true,
    )
end

Radius = Data_Radius_List[Index]
# if Index < 3
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
Radius = Implicit_Radius_List[Index]

Cheb_Order = 2
Cheb_Intervals = 120
Polar_Order = 17
MIP, XIP, Torus, E_WW_Full, Latent_Data, E_ENC, AA, Valid_Ind = Extract_Manifold_Embedding(
    MTF,
    XTF,
    Index,
    Data_Decomp_T,
    Encoded_Phase_T;
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
MTF_Cache, DATA_Backbone, DATA_Error_Curves, Data_Max = Data_Result(
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
MTF_Cache, Data_Max, TEST_Error_Curves = Data_Error(
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
    Model_IC = true,
)

fig = Create_Plot()
Plot_Backbone_Curves!(fig, ODE_Backbone, Data_Max; Label = "ODE", Color = Makie.wong_colors()[2])
Plot_Backbone_Curves!(fig, MAP_Backbone, Data_Max; Label = "MAP", Color = Makie.wong_colors()[3])
Plot_Backbone_Curves!(fig, DATA_Backbone, Data_Max; Label = "Data", Color = Makie.wong_colors()[1])
Plot_Error_Curves!(fig, DATA_Error_Curves, Data_Max; Color = Makie.wong_colors()[1])
Plot_Error_Curves!(fig, TEST_Error_Curves, Data_Max; Color = Makie.wong_colors()[2])
Plot_Error_Trace(fig, Index, Error_Trace, Test_Trace)
Annotate_Plot!(fig)
###

CairoMakie.activate!(type = "svg")
save("FIGURE-$(Name)-$(VER)-I$(Index).svg", fig)
GLMakie.activate!()
push!(fig1_List, fig)

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

display(fig2)
CairoMakie.activate!(type = "svg")
save("FIGURE-LATENT-$(Name)-$(VER)-I$(Index).svg", fig2)
GLMakie.activate!()
push!(fig2_List, fig2)

fig3 = Figure()

# data = JLSO.load("RAWDATA-$(Name)-$(RAWDATAVER).bson")
# List_of_Data        = data[:List_of_Data_T]
# List_of_Phases      = data[:List_of_Phases_T]
#
# E_WW_Latent_SS, E_TT, Valid = Latent_To_Manifold(PPM, PPX, MTF, XTF, Index, MTF_Cache.Parts[Index].Model_Cache.Values, Encoded_Phase; Transformation=Data_Decoder ./ reshape(Data_Scale,1,1,:))
# E_WW_Latent = E_WW_Latent_SS + Steady_State * Encoded_Phase
Repr_Dims = 1:State_Dimension
Select_Test = 1
E_WW_Latent = E_WW_Full[Repr_Dims, :]
E_WW_Latent += Steady_State[Repr_Dims, :] * Encoded_Phase_T
Full_Decoder = Data_Decoder[Repr_Dims, :, :] ./ reshape(Data_Scale, 1, 1, :)
@tullio Data_Raw[j, k] := Full_Decoder[j, p, q] * Data_Decomp_T[q, k] * Encoded_Phase_T[p, k]
Data_Raw += Steady_State[Repr_Dims, :] * Encoded_Phase_T

X_Axis = range(0, step = ct.Time_Step, length = size(Data_Raw, 2))
X_Axis_Model = range(0, step = ct.Time_Step, length = size(E_WW_Latent, 2))
#
Invalid = findall(.!Valid_Ind)
#
Rows = round(Int, sqrt(size(E_WW_Latent, 1)))
let it = 0
    for k = 1:size(E_WW_Latent, 1)
        ax = Makie.Axis(
            fig3[1+div(it, Rows), 1+mod(it, Rows)],
            xlabel = L"$t$ [s]",
            ylabel = L"z_{%$k}",
        )
        lines!(ax, X_Axis, Data_Raw[k, :], color = Makie.wong_colors()[1])
        it += 1
    end
end
#
let it = 0
    for k = 1:size(E_WW_Latent, 1)
        ax = content(fig3[1+div(it, Rows), 1+mod(it, Rows)])
        lines!(ax, X_Axis_Model, E_WW_Latent[k, :], color = Makie.wong_colors()[2])
        scatter!(
            ax,
            X_Axis_Model[Invalid],
            E_WW_Latent[k, Invalid],
            color = Makie.wong_colors()[3],
        )
        it += 1
    end
end
CairoMakie.activate!(type = "svg")
save(
    "FIGURE-MANIF-$(Name)-$(VER)-I$(Index).svg",
    fig3,
)
GLMakie.activate!()
push!(fig3_List, fig3)
JLSO.save(
    "CURVES-$(Name)-$(VER)-I$(Index).bson",
    :ODE_Backbone => ODE_Backbone,
    :MAP_Backbone => MAP_Backbone,
    :DATA_Backbone => DATA_Backbone,
    :DATA_Error_Curves => DATA_Error_Curves,
    :TEST_Error_Curves => TEST_Error_Curves,
    :Error_Trace => Error_Trace,
    :Test_Trace => Test_Trace,
    :Data_Raw => Data_Raw,
    :Data_Reconstructed => E_WW_Latent,
    :Valid_Id => Valid_Ind,
    :Invalid_Id => Invalid,
    :Latent_Model => MTF_Cache.Parts[Index].Model_Cache.Values,
    :Latent_Data => MTF_Cache.Parts[Index].Latent_Data,
    :Index_List => Index_List_T,
    :Time_Step => Time_Step,
)
end # Index in 1:2
end
