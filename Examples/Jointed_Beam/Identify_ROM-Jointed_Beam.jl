Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels
using MAT
using StatsBase

Suffix = ["0_0Nm" "1_0Nm" "2_1Nm" "3_1Nm"]
S_Select = 2
Name = "Jointed_Beam-$(Suffix[S_Select])"
VER = "V1"
DATAVER = "V1"
Generate = false
Process = false

# autonomous
Skew_Dimension = 1 # must be and odd number

Forcing_Grid = Fourier_Grid(Skew_Dimension)

if Generate
    dd = JLSO.load("DATA-Jointed_Beam-$(Suffix[S_Select]).bson")

    Time_Step = mean([mean(tr[2:end] - tr[1:end-1]) for tr in dd[:Time_Code]])
    Raw_Signal = [reshape(tr, 1, :) for tr in dd[:Signals]]
    Raw_Phase = [zeros(1, size(sig, 2)) for sig in Raw_Signal]

    Signals, Phases_Angle = Delay_Embed(Raw_Signal, Raw_Phase; delay = 18, skip = 1)
    Phases = [Fourier_Interpolate(Fourier_Grid(1), vec(ph)) for ph in Phases_Angle]
    Sel_Train = [1; 3; 5; 6; 7; 8]
    Sel_Test = [2; 4]
    List_of_Data = Signals[Sel_Train]
    List_of_Phases = Phases[Sel_Train]
    List_of_Data_T = Signals[Sel_Test]
    List_of_Phases_T = Phases[Sel_Test]

    @show "RAWDATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "RAWDATA-$(Name)-$(DATAVER).bson",
        #         :Parameters         => Parameters,
        :Time_Step => Time_Step,
        :List_of_Data => List_of_Data,
        :List_of_Phases => List_of_Phases,
        :List_of_Data_T => List_of_Data_T,
        :List_of_Phases_T => List_of_Phases_T,
    )
else
    data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
    #     Parameters          = data[:Parameters]
    Time_Step = data[:Time_Step]
    List_of_Data = data[:List_of_Data]
    List_of_Phases = data[:List_of_Phases]
    List_of_Data_T = data[:List_of_Data_T]
    List_of_Phases_T = data[:List_of_Phases_T]
end

if Process
    Index_List, Data, Encoded_Phase =
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 1180)
    Index_List_T, Data_T, Encoded_Phase_T =
        Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 1180)

    # the number of chopped trajectories
    Scaling = 1 ./ (0.01 .+ sqrt.(sum(Data .^ 2, dims = 1)))
    Steady_State, Linear_Model, SH = Estimate_Linear_Model(
        Index_List,
        Data,
        Encoded_Phase,
        Scaling;
        Iterations = 0,
        Order = 1,
    )
    # filtering up to 2 harmonics
    Linear_Model_Filtered = Filter_Linear_Model(Linear_Model, Forcing_Grid, 1)
    Decomp_Zero = Create_Linear_Decomposition(
        Linear_Model_Filtered,
        SH;
        Time_Step = Time_Step,
        Reduce = true,
        Align = true,
        By_Eigen = true,
    )
    Decomp = Select_Bundles_By_Energy(
        Index_List,
        Data,
        Encoded_Phase,
        Steady_State,
        SH,
        Decomp_Zero;
        How_Many = 6,
    )
    Data_Decomp, _ = Decompose_Data(
        Index_List,
        Data,
        Encoded_Phase,
        Steady_State,
        SH,
        Decomp.Data_Encoder,
    )
    Data_Decomp_T, _ = Decompose_Data(
        Index_List_T,
        Data_T,
        Encoded_Phase_T,
        Steady_State,
        SH,
        Decomp.Data_Encoder,
    )
    # Scaling
    Data_Scale = Decomposed_Data_Scaling(Data_Decomp, Decomp.Bundles)
    Data_Decomp .*= Data_Scale
    Data_Decomp_T .*= Data_Scale

    @show "DATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "DATA-$(Name)-$(DATAVER).bson",
        #         :Parameters      => Parameters,
        :Time_Step => Time_Step,
        :Index_List => Index_List,
        :Data_Decomp => Data_Decomp,
        :Encoded_Phase => Encoded_Phase,
        :Index_List_T => Index_List_T,
        :Data_Decomp_T => Data_Decomp_T,
        :Encoded_Phase_T => Encoded_Phase_T,
        :Decomp => Decomp,
        :Steady_State => Steady_State,
        :Linear_Model => Linear_Model_Filtered,
        :SH => SH,
        :Data_Scale => Data_Scale,
    )
else
    data = JLSO.load("DATA-$(Name)-$(DATAVER).bson")
    #     Parameters      = data[:Parameters]
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
end

MTFP = Multi_Foliation_Problem(
    Index_List,
    Data_Decomp,
    Encoded_Phase,
    Selection = ([1; 2], [3; 4], [5; 6], [7; 8], [9; 10], [11; 12]),
    Model_Orders = (5, 5, 5, 5, 3, 3),
    Encoder_Orders = (5, 5, 5, 5, 3, 3),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 16,
    Scaling_Parameter = 2^(-4),
    Initial_Scaling_Parameter = 2^(-4),
    Scaling_Order = Linear_Scaling,
    node_ratio = 0.8,
    leaf_ratio = 0.8,
    max_rank = 16,
    Linear_Type = (
        Encoder_Array_Stiefel,
        Encoder_Array_Stiefel,
        Encoder_Array_Stiefel,
        Encoder_Array_Stiefel,
        Encoder_Array_Stiefel,
        Encoder_Array_Stiefel,
    ),
    Nonlinear_Type = (
        Encoder_Compressed_Latent_Linear,
        Encoder_Compressed_Latent_Linear,
        Encoder_Compressed_Latent_Linear,
        Encoder_Compressed_Latent_Linear,
        Encoder_Compressed_Latent_Linear,
        Encoder_Compressed_Latent_Linear,
    ),
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
