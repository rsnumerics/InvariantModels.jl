Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels

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
VER = "Autonomous"
DATAVER = "Autonomous"
# VER = "Forced"
# DATAVER = "Forced"
Generate = true
Process = true

# autonomous
Skew_Dimension = 1 # must be and odd number
Training_Trajectories = 16
Testing_Trajectories = 2
Forcing_Amplitude = 0.0
Omega_ODE = sqrt(0.4)
Trajectory_Length = 1000

# forced
# Skew_Dimension = 17 # must be an odd number
# Training_Trajectories = 16
# Testing_Trajectories = 1
# Forcing_Amplitude = 0.1
# Omega_ODE = sqrt(0.4)
# Trajectory_Length = 1000

Forcing_Grid = Fourier_Grid(Skew_Dimension)
Parameters = (
    Weights = sin.(Forcing_Grid),
    Forcing_Grid = Forcing_Grid,
    Amplitude = Forcing_Amplitude,
    Omega = Omega_ODE,
)
Time_Step = 2 * pi / 17 / Omega_ODE

if Generate
#     AA = randn(9, Training_Trajectories)
#     IC_x_Train =
#         0.4 * # 0.3 for forced, 0.4 for autonomous
#         AA *
#         diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
#         diagm(0.4 .+ 0.7 .* rand(size(AA, 2)))
#     AA = randn(9, Testing_Trajectories)
#     IC_x_Test =
#         0.95 *
#         0.4 * # 0.3 for forced, 0.4 for autonomous
#         AA *
#         diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
#         diagm(0.4 .+ 0.7 .* rand(size(AA, 2)))
#
#     function Random_Phase(Skew_Dimension, Trajectories)
#         IC_Alpha = zeros(Skew_Dimension, Trajectories)
#         if Skew_Dimension >= Trajectories
#             Start_Phase = randperm(Skew_Dimension)[1:Trajectories]
#         else
#             Start_Phase = vcat(
#                 repeat(randperm(Skew_Dimension), div(Trajectories, Skew_Dimension)),
#                 randperm(Skew_Dimension)[1:mod(Trajectories, Skew_Dimension)],
#             )
#         end
#         for k in eachindex(Start_Phase)
#             IC_Alpha[Start_Phase[k], k] = 1
#         end
#         return IC_Alpha
#     end
#
#     IC_Alpha_Train = Random_Phase(Skew_Dimension, Training_Trajectories)
#     IC_Alpha_Test = Random_Phase(Skew_Dimension, Testing_Trajectories)

    IC_Dict = JLSO.load("ICS-$(Name)-$(DATAVER).bson")
    IC_x_Train = IC_Dict[:IC_x_Train]
    IC_x_Test = IC_Dict[:IC_x_Test]
    IC_Alpha_Train = IC_Dict[:IC_Alpha_Train]
    IC_Alpha_Test = IC_Dict[:IC_Alpha_Test]

    IC_Force = [0.0]
    List_of_Data, List_of_Phases = Generate_From_ODE(
        Car_Follow_VF!,
        Car_Follow_Forcing!,
        Car_Follow_Forcing_Matrix!,
        Parameters,
        Time_Step,
        IC_x_Train,
        IC_Force,
        IC_Alpha_Train,
        ones(Int, Training_Trajectories) * Trajectory_Length,
    )
    List_of_Data_T, List_of_Phases_T = Generate_From_ODE(
        Car_Follow_VF!,
        Car_Follow_Forcing!,
        Car_Follow_Forcing_Matrix!,
        Parameters,
        Time_Step,
        IC_x_Test,
        IC_Force,
        IC_Alpha_Test,
        ones(Int, Testing_Trajectories) * Trajectory_Length,
    )
#     @show "RAWDATA-$(Name)-$(DATAVER).bson"
#     JLSO.save(
#         "RAWDATA-$(Name)-$(DATAVER).bson",
#         :Parameters         => Parameters,
#         :Time_Step          => Time_Step,
#         :List_of_Data       => List_of_Data,
#         :List_of_Phases     => List_of_Phases,
#         :List_of_Data_T     => List_of_Data_T,
#         :List_of_Phases_T   => List_of_Phases_T,
#     )
else
    data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
    Parameters          = data[:Parameters]
    Time_Step           = data[:Time_Step]
    List_of_Data        = data[:List_of_Data]
    List_of_Phases      = data[:List_of_Phases]
    List_of_Data_T      = data[:List_of_Data_T]
    List_of_Phases_T    = data[:List_of_Phases_T]

    # making it reproducible
#     IC_x_Train = hcat([da[:,1] for da in data[:List_of_Data]]...)
#     IC_x_Test = hcat([da[:,1] for da in data[:List_of_Data_T]]...)
#     IC_Alpha_Train = hcat([da[:,1] for da in data[:List_of_Phases]]...)
#     IC_Alpha_Test = hcat([da[:,1] for da in data[:List_of_Phases_T]]...)
#     JLSO.save("ICS-$(Name)-$(DATAVER).bson", :IC_x_Train => IC_x_Train, :IC_x_Test => IC_x_Test, :IC_Alpha_Train => IC_Alpha_Train, :IC_Alpha_Test => IC_Alpha_Test, format = :bson, compression = :none)
end

if Process
    Index_List, Data, Encoded_Phase =
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 499)
    Index_List_T, Data_T, Encoded_Phase_T =
        Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 499)

    # the number of chopped trajectories
    # Number_Of_Trajectories = length(Index_List) - 1
    Scaling = 1 ./ ((2^-8) .+ sqrt.(sum(Data .^ 2, dims = 1)))
    Steady_State, Linear_Model, SH = Estimate_Linear_Model(
        Index_List,
        Data,
        Encoded_Phase,
        Scaling;
        Iterations=0,
        Order=1,
    )
    # filtering up to 2 harmonics
    Linear_Model_Filtered = Filter_Linear_Model(Linear_Model, Forcing_Grid, 1)
    Decomp = Create_Linear_Decomposition(
        Linear_Model_Filtered,
        SH;
        Time_Step=Time_Step,
        Reduce=true,
        Align=true,
        By_Eigen=true,
    )
    Data_Decomp, _ = Decompose_Data(Index_List, Data, Encoded_Phase, Steady_State, SH, Decomp.Data_Encoder)
    Data_Decomp_T, _ = Decompose_Data(Index_List_T, Data_T, Encoded_Phase_T, Steady_State, SH, Decomp.Data_Encoder)
    # Scaling
    Data_Scale = Decomposed_Data_Scaling(Data_Decomp, Decomp.Bundles)
    Data_Decomp .*= Data_Scale
    Data_Decomp_T .*= Data_Scale

    @show "DATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "DATA-$(Name)-$(DATAVER).bson",
        :Parameters      => Parameters,
        :Time_Step       => Time_Step,
        :Index_List      => Index_List,
        :Data_Decomp     => Data_Decomp,
        :Encoded_Phase   => Encoded_Phase,
        :Index_List_T    => Index_List_T,
        :Data_Decomp_T   => Data_Decomp_T,
        :Encoded_Phase_T => Encoded_Phase_T,
        :Decomp          => Decomp,
        :Steady_State    => Steady_State,
        :Linear_Model    => Linear_Model_Filtered,
        :SH              => SH,
        :Data_Scale      => Data_Scale,
    )
else
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
end

MTFP = Multi_Foliation_Problem(
    Index_List,
    Data_Decomp,
    Encoded_Phase,
    Selection = ([1; 2], [3; 4; 5; 6; 7; 8; 9]),
    Model_Orders = (3, 1),
    Encoder_Orders = (2, 3),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 32,
    Scaling_Parameter = 2^(-10),
    Initial_Scaling_Parameter = 2^(-4),
    Scaling_Order = Linear_Scaling,
    node_ratio = 1.0,
    leaf_ratio = 1.0,
    max_rank = 20,
    Linear_Type = (Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
    Nonlinear_Type = (Encoder_Compressed_Latent_Linear, Encoder_Compressed_Local),
    Name = "MTF-$(Name)-$(VER)",
    Time_Step = Time_Step,
)
MTFP_Test = Multi_Foliation_Test_Problem(
    MTFP,
    Index_List_T,
    Data_Decomp_T,
    Encoded_Phase_T;
    Initial_Scaling_Parameter = 2^(-2),
)
Optimise!(
    MTFP,
    MTFP_Test;
    Model_Iterations = 16,
    Encoder_Iterations = 8,
    Steps = 1000,
    Gradient_Ratio = 2^(-7),
    Gradient_Stop = 2^(-29),
)

end
