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

Name = "Brake_Reuss"
VER = "V3"
DATAVER = "V1"
Generate = false
Process = false

# autonomous
Skew_Dimension = 1 # must be and odd number

Forcing_Grid = Fourier_Grid(Skew_Dimension)

if Generate
    dd = matread("data.mat")
    tt = dd["data_BRB"]["TimeACC"]
    Time_Step = mean(tt[2:end] - tt[1:end-1])
    start_index = findfirst(vec(tt) .>= 0)
    Raw_Signal = dd["data_BRB"]["AccelerationACC"][:, start_index:272000]

    Raw_Phase = zeros(1, size(Raw_Signal, 2))

    List_of_Data, Phases_Angle =
        Delay_Embed([Raw_Signal], [Raw_Phase]; delay = 64, skip = 1)
    List_of_Phases = [Fourier_Interpolate(Fourier_Grid(1), vec(ph)) for ph in Phases_Angle]

    @show "RAWDATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "RAWDATA-$(Name)-$(DATAVER).bson",
        #         :Parameters         => Parameters,
        :Time_Step => Time_Step,
        :List_of_Data => List_of_Data,
        :List_of_Phases => List_of_Phases,
        #         :List_of_Data_T     => List_of_Data_T,
        #         :List_of_Phases_T   => List_of_Phases_T,
    )
else
    data = JLSO.load("RAWDATA-$(Name)-$(DATAVER).bson")
    #     Parameters          = data[:Parameters]
    Time_Step = data[:Time_Step]
    List_of_Data = data[:List_of_Data]
    List_of_Phases = data[:List_of_Phases]
    #     List_of_Data_T      = data[:List_of_Data_T]
    #     List_of_Phases_T    = data[:List_of_Phases_T]
end

if Process
    Index_List, Data, Encoded_Phase =
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 2400)
    #     Index_List_T, Data_T, Encoded_Phase_T =
    #         Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 499)

    # the number of chopped trajectories
    # Number_Of_Trajectories = length(Index_List) - 1
    Scaling = 1 ./ (0.0001 .+ sqrt.(sum(Data .^ 2, dims = 1)))
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
        How_Many = 4,
    )
    Data_Decomp, _ = Decompose_Data(
        Index_List,
        Data,
        Encoded_Phase,
        Steady_State,
        SH,
        Decomp.Data_Encoder,
    )
    #     Data_Decomp_T, _ = Decompose_Data(Index_List_T, Data_T, Encoded_Phase_T, Steady_State, SH, Decomp.Data_Encoder)
    # Scaling
    Data_Scale = Decomposed_Data_Scaling(Data_Decomp, Decomp.Bundles)
    Data_Decomp .*= Data_Scale
    #     Data_Decomp_T .*= Data_Scale

    @show "DATA-$(Name)-$(DATAVER).bson"
    JLSO.save(
        "DATA-$(Name)-$(DATAVER).bson",
        #         :Parameters      => Parameters,
        :Time_Step => Time_Step,
        :Index_List => Index_List,
        :Data_Decomp => Data_Decomp,
        :Encoded_Phase => Encoded_Phase,
        #         :Index_List_T    => Index_List_T,
        #         :Data_Decomp_T   => Data_Decomp_T,
        #         :Encoded_Phase_T => Encoded_Phase_T,
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
    #     Index_List_T    = data[:Index_List_T]
    #     Data_Decomp_T   = data[:Data_Decomp_T]
    #     Encoded_Phase_T = data[:Encoded_Phase_T]
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
    Selection = ([1; 2], [3; 4; 5; 6; 7; 8]),
    Model_Orders = (11, 2),
    Encoder_Orders = (2, 9),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 16,
    Scaling_Parameter = 2^(-3),
    Initial_Scaling_Parameter = 2^(-2),
    Scaling_Order = Linear_Scaling,
    node_ratio = 1.0,
    leaf_ratio = 1.0,
    max_rank = 24,
    Linear_Type = (Encoder_Array_Stiefel, Encoder_Array_Stiefel),
    Nonlinear_Type = (Encoder_Compressed_Latent_Linear, Encoder_Compressed_Local),
    Name = "MTF-$(Name)-$(VER)",
    Time_Step = Time_Step,
    Train_Model = false,
)
# MTFP_Test = Multi_Foliation_Test_Problem(
#     MTFP,
#     Index_List_T,
#     Data_Decomp_T,
#     Encoded_Phase_T;
#     Initial_Scaling_Parameter = 2^(-2),
# )
Optimise!(
    MTFP,
    nothing; #MTFP_Test;
    Model_Iterations = 16,
    Encoder_Iterations = 8,
    Steps = 1000,
    Gradient_Ratio = 2^(-7),
    Gradient_Stop = 2^(-29),
)

end
