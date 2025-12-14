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

Name = "Foil_Auto"
VER = "V4-1F"
DATAVER = "V4-1F"
RAWDATAVER = "V4-1"

data = JLSO.load("DATA-$(Name)-$(DATAVER).bson")
# Parameters      = data[:Parameters]
Time_Step = data[:Time_Step]
Index_List = data[:Index_List]
Data_Decomp = data[:Data_Decomp]
Encoded_Phase = data[:Encoded_Phase]
Index_List_T = data[:Index_List_T]
Data_Decomp_T = data[:Data_Decomp_T]
Encoded_Phase_T = data[:Encoded_Phase_T]
Decomp = data[:Decomp]
Steady_State = data[:Steady_State]
Linear_Model = data[:Linear_Model]
SH = data[:SH]
Data_Scale = data[:Data_Scale]
#
Data_Encoder = Decomp.Data_Encoder
Data_Decoder = Decomp.Data_Decoder
# Omega_ODE         = Parameters.Omega
# Forcing_Amplitude = Parameters.Amplitude
State_Dimension = size(Data_Decomp, 1)
Skew_Dimension = size(Encoded_Phase, 1)

dd = JLSO.load("MTF-$(Name)-$(VER).bson")
MTF = dd[:MTF]
XTF = dd[:XTF]
Error_Trace = dd[:Train_Error_Trace]
MTF_Test = dd[:Test_MTF]
XTF_Test = dd[:Test_XTF]
Test_Trace = dd[:Test_Error_Trace]

for k in eachindex(XTF_Test.x)
    XTF_Test.x[k].x[1].WW .= XTF.x[k].x[1].WW
end

fig1_List = []
fig2_List = []
fig3_List = []
for Index in 1:3

# V1
# Data_Radius_List = (0.95, 0.2, 0.15, 0.15, 1.0)
# Implicit_Radius_List = (0.55, 1.0, 1.0, 1.0, 1.0)
# V2
# Data_Radius_List = (0.5, 0.6, 0.2, 0.5, 1.0)
# Implicit_Radius_List = (0.8, 0.8, 0.9, 0.8, 1.0)
# V3
Data_Radius_List = (0.2, 0.2, 0.2, 0.2, 1.0)
Implicit_Radius_List = (0.8, 1.0, 0.8, 0.8, 1.0)

#
Radius = Data_Radius_List[Index]
Cheb_Order = 2
Cheb_Intervals = 200
Polar_Order = 17

PPM, PPX = Find_DATA_Manifold(
    MTF,
    XTF,
    SH,
    Index;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    Transformation = Data_Decoder[1:3, :, :],
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)

Radius = Implicit_Radius_List[Index]

Cheb_Order = 2
Cheb_Intervals = 120
Polar_Order = 17
MIP, XIP, Torus, E_WW_Full, Latent_Data, E_ENC, AA, Valid_Ind = Extract_Manifold_Embedding(
    MTF_Test,
    XTF_Test,
    Index,
    Data_Decomp_T,
    Encoded_Phase_T;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    #     Output_Transformation = Data_Encoder,
    Output_Inverse_Transformation = Data_Decoder,
    Output_Scale = vec(Data_Scale),
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)

###
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
    Transformation = Data_Decoder[1:3, :, :] ./ reshape(Data_Scale, 1, 1, :),
    Time_Step = Time_Step,
    Hz = true,
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
    Transformation = Data_Decoder[1:3, :, :] ./ reshape(Data_Scale, 1, 1, :),
    Model_IC = false,
)

fig = Create_Plot()
# Plot_Backbone_Curves!(fig, ODE_Backbone, Data_Max; Label = "ODE", Color = Makie.wong_colors()[2])
# Plot_Backbone_Curves!(fig, MAP_Backbone, Data_Max; Label = "MAP", Color = Makie.wong_colors()[3])
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
Select_Test = 1
E_WW_Latent = E_WW_Full[1:3, :]
E_WW_Latent += Steady_State[1:3, :] * Encoded_Phase_T
Full_Decoder = Data_Decoder[1:3, :, :] ./ reshape(Data_Scale, 1, 1, :)
@tullio Data_Raw[j, k] := Full_Decoder[j, p, q] * Data_Decomp_T[q, k] * Encoded_Phase_T[p, k]
Data_Raw += Steady_State[1:3, :] * Encoded_Phase_T

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
    #     :ODE_Backbone => ODE_Backbone,
    #     :MAP_Backbone => MAP_Backbone,
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

end # for Index in ...

end
