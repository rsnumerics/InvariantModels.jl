Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels

include("sys10dimvectorfield.jl")

Name = "Ten_Dimensional"
VER = "V2"
DATAVER = "V1"
Generate = false
Process = false

# autonomous
Skew_Dimension = 1 # must be and odd number
Training_Trajectories = 5
Testing_Trajectories = 1
Forcing_Amplitude = 0.0
Omega_ODE = 2^(1 / 4)
Trajectory_Length = 40000

# forced
# Skew_Dimension = 19 # must be and odd number
# Training_Trajectories = 16
# Testing_Trajectories = 1
# Forcing_Amplitude = 0.04
# Omega_ODE = 2^(1/4)
# Trajectory_Length = 1350

Forcing_Grid = Fourier_Grid(Skew_Dimension)
Parameters = (
    Forcing_Grid = Forcing_Grid,
)
Time_Step = 0.07

if Generate
    AA = randn(10, Training_Trajectories)
    IC_x_Train =
        0.6 *
        AA *
        diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
        diagm(0.95 .+ 0.1 .* rand(size(AA, 2)))
    AA = randn(10, Testing_Trajectories)
    IC_x_Test =
        0.95 *
        0.6 *
        AA *
        diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
        diagm(0.95 .+ 0.1 .* rand(size(AA, 2)))

    function Random_Phase(Skew_Dimension, Trajectories)
        IC_Alpha = zeros(Skew_Dimension, Trajectories)
        if Skew_Dimension >= Trajectories
            Start_Phase = randperm(Skew_Dimension)[1:Trajectories]
        else
            Start_Phase = vcat(
                repeat(randperm(Skew_Dimension), div(Trajectories, Skew_Dimension)),
                randperm(Skew_Dimension)[1:mod(Trajectories, Skew_Dimension)],
            )
        end
        for k in eachindex(Start_Phase)
            IC_Alpha[Start_Phase[k], k] = 1
        end
        return IC_Alpha
    end

    IC_Alpha_Train = Random_Phase(Skew_Dimension, Training_Trajectories)
    IC_Alpha_Test = Random_Phase(Skew_Dimension, Testing_Trajectories)

    IC_Force = [0.0]
    List_of_Data, List_of_Phases = Generate_From_ODE(
        Ten_Dimensional_VF!,
        Ten_Dimensional_Forcing!,
        Ten_Dimensional_Forcing_Matrix!,
        Parameters,
        Time_Step,
        IC_x_Train,
        IC_Force,
        IC_Alpha_Train,
        ones(Int, Training_Trajectories) * Trajectory_Length,
    )
    List_of_Data_T, List_of_Phases_T = Generate_From_ODE(
        Ten_Dimensional_VF!,
        Ten_Dimensional_Forcing!,
        Ten_Dimensional_Forcing_Matrix!,
        Parameters,
        Time_Step,
        IC_x_Test,
        IC_Force,
        IC_Alpha_Test,
        ones(Int, Testing_Trajectories) * Trajectory_Length,
    )
    @show "RAWDATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "RAWDATA-$(Name)-$(DATAVER).bson",
        :Parameters         => Parameters,
        :Time_Step          => Time_Step,
        :List_of_Data       => List_of_Data,
        :List_of_Phases     => List_of_Phases,
        :List_of_Data_T     => List_of_Data_T,
        :List_of_Phases_T   => List_of_Phases_T,
    )
else
    data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
    Parameters          = data[:Parameters]
    Time_Step           = data[:Time_Step]
    List_of_Data        = data[:List_of_Data]
    List_of_Phases      = data[:List_of_Phases]
    List_of_Data_T      = data[:List_of_Data_T]
    List_of_Phases_T    = data[:List_of_Phases_T]
end

if Process
    Index_List, Data, Encoded_Phase =
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 1999)
    Index_List_T, Data_T, Encoded_Phase_T =
        Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 1999)

    # the number of chopped trajectories
    # Number_Of_Trajectories = length(Index_List) - 1
    Scaling = 1 ./ ((2^-12) .+ sqrt.(sum(Data .^ 2, dims = 1)))
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
    Selection=([1;2],[3;4],[5;6],[7;8],[9;10]),
    Model_Orders=(5,5,5,5,5,),
    Encoder_Orders=(5,5,5,5,5,),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 32,
    Scaling_Parameter = 2^(-2),
    Initial_Scaling_Parameter = 2^(-2),
    Scaling_Order = Linear_Scaling,
    node_ratio = 0.8,
    leaf_ratio = 1.0,
    max_rank = 24,
    Linear_Type=(Encoder_Mean_Stiefel, Encoder_Mean_Stiefel, Encoder_Mean_Stiefel, Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
    Nonlinear_Type=(Encoder_Compressed_Latent_Linear, Encoder_Compressed_Latent_Linear, Encoder_Compressed_Latent_Linear, Encoder_Compressed_Latent_Linear, Encoder_Compressed_Latent_Linear),
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
