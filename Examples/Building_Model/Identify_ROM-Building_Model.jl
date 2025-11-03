Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels
using StatsBase

Name = "Building_Model"
VER = "V3"
DATAVER = "V3"
Process = true

# autonomous

data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
Time_Step           = data[:Time_Step]
List_of_Data        = data[:List_of_Data]
List_of_Phases      = data[:List_of_Phases]
List_of_Data_T      = data[:List_of_Data_T]
List_of_Phases_T    = data[:List_of_Phases_T]

Skew_Dimension = size(List_of_Phases[1], 1) # must be and odd number
Forcing_Grid = Fourier_Grid(Skew_Dimension)

if Process
    Index_List, Data, Encoded_Phase =
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 1200)
    Index_List_T, Data_T, Encoded_Phase_T =
        Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 1200)

    # the number of chopped trajectories
    Scaling = 1 ./ ( 0.01 .+ sqrt.(sum(Data .^ 2, dims=1)))
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
    Decomp_Zero = Create_Linear_Decomposition(
        Linear_Model_Filtered,
        SH;
        Time_Step=Time_Step,
        Reduce=true,
        Align=true,
        By_Eigen=true,
    )
    Decomp = Select_Bundles_By_Energy(Index_List, Data, Encoded_Phase, Steady_State, SH, Decomp_Zero; How_Many=5, Time_Step=Time_Step)

    Data_Decomp, _ = Decompose_Data(Index_List, Data, Encoded_Phase, Steady_State, SH, Decomp.Data_Encoder)
    Data_Decomp_T, _ = Decompose_Data(Index_List_T, Data_T, Encoded_Phase_T, Steady_State, SH, Decomp.Data_Encoder)
    # Scaling
    Data_Scale = Decomposed_Data_Scaling(Data_Decomp, Decomp.Bundles)
    Data_Decomp .*= Data_Scale
    Data_Decomp_T .*= Data_Scale

    @show "DATA-$(Name)-$(VER).bson"
    JLSO.save(
        "DATA-$(Name)-$(VER).bson",
#         :Parameters      => Parameters,
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
    data = JLSO.load("DATA-$(Name)-$(VER).bson")
#     Parameters      = data[:Parameters]
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
    Selection = ([1;2],[3;4],[5;6],[7;8],[9;10]),
    Model_Orders = (5, 5, 5, 5, 5),
    Encoder_Orders = (5, 5, 5, 5, 5),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 16,
    Scaling_Parameter = 2^(-3),
    Initial_Scaling_Parameter = 2^(-3),
    Scaling_Order = Linear_Scaling,
    node_ratio = 0.8,
    leaf_ratio = 0.8,
    max_rank = 16,
    Linear_Type = (Encoder_Array_Stiefel, Encoder_Array_Stiefel, Encoder_Array_Stiefel, Encoder_Array_Stiefel, Encoder_Array_Stiefel),
    Nonlinear_Type = (Encoder_Compressed_Full, Encoder_Compressed_Full, Encoder_Compressed_Full, Encoder_Compressed_Full, Encoder_Compressed_Full),
    Name = "MTF-$(Name)-$(VER)",
    Time_Step = Time_Step,
)
MTFP_Test = Multi_Foliation_Test_Problem(
    MTFP,
    Index_List_T,
    Data_Decomp_T,
    Encoded_Phase_T;
    Initial_Scaling_Parameter = 2^(-3),
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
