Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels
using StatsBase

Name = "Foil_Forced"
VER = "V4-29F"
DATAVER = "V4-29"
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
    if isfile("LINEARMODEL-$(Name)-$(VER).bson")
        dd = JLSO.load("LINEARMODEL-$(Name)-$(VER).bson")
        Steady_State_R = dd[:Steady_State]
        Linear_Model_R = dd[:Linear_Model]
        SH_R = dd[:SH]
    else
        # re-creating phase with coarser discretisation
        Phases_Rough = [Fourier_Interpolate(Fourier_Grid(5), vec(ph)) for ph in data[:List_of_Angles]]
        Index_List_R, Data_R, Encoded_Phase_R =
            Chop_And_Stitch(List_of_Data, Phases_Rough; maxlen = 1200)
        Scaling = ones(size(Data_R, 2))
        Steady_State_R, Linear_Model_R, SH_R = Estimate_Linear_Model(
            Index_List_R,
            Data_R,
            Encoded_Phase_R,
            Scaling;
            Iterations=0,
            Order=1,
        )
        JLSO.save("LINEARMODEL-$(Name)-$(VER).bson", :Steady_State => Steady_State_R, :Linear_Model => Linear_Model_R, :SH => SH_R)
    end

    # filtering up to 2 harmonics
    Linear_Model_Filtered_R = Filter_Linear_Model(Linear_Model_R, Fourier_Grid(size(SH_R, 1)), 0)
    Decomp_Zero_R = Create_Linear_Decomposition(
        Linear_Model_Filtered_R,
        SH_R;
        Time_Step=Time_Step,
        Reduce=false,
        Align=true,
        By_Eigen=true,
        maxiter=18000,
    )
    # interpolating the whole decomposition back
    Grid = Fourier_Grid(size(Encoded_Phase, 1))
    FITP = Fourier_Interpolate(Fourier_Grid(size(SH_R, 1)), Grid)
    @tullio Linear_Model[i, k, l] := Linear_Model_R[i, j, l] * FITP[j, k]
    @tullio Linear_Model_Filtered[i, k, l] := Linear_Model_Filtered_R[i, j, l] * FITP[j, k]
    @tullio Steady_State[i, k] := Steady_State_R[i, j] * FITP[j, k]
    @tullio Unreduced_Model[i, j, k] := Decomp_Zero_R.Unreduced_Model[i, p, k] * FITP[p, j]
    @tullio Data_Encoder[i, j, k] := Decomp_Zero_R.Data_Encoder[i, p, k] * FITP[p, j]
    @tullio Data_Decoder[i, j, k] := Decomp_Zero_R.Data_Decoder[i, p, k] * FITP[p, j]
    @tullio Reduced_Model[i, j, k] := Decomp_Zero_R.Reduced_Model[i, p, k] * FITP[p, j]
    @tullio Reduced_Encoder[i, j, k] := Decomp_Zero_R.Reduced_Encoder[i, p, k] * FITP[p, j]
    @tullio Reduced_Decoder[i, j, k] := Decomp_Zero_R.Reduced_Decoder[i, p, k] * FITP[p, j]
    SH = InvariantModels.Find_Shift(Index_List, Data, Encoded_Phase)
    Decomp_Zero = (
        Unreduced_Model = Unreduced_Model,
        Data_Encoder = Data_Encoder,
        Data_Decoder = Data_Decoder,
        Bundles = Decomp_Zero_R.Bundles,
        Reduced_Model = Reduced_Model,
        Reduced_Encoder = Reduced_Encoder,
        Reduced_Decoder = Reduced_Decoder,
    )
    Bundle_Indices = [1;4;3]
    Select = vcat(Decomp_Zero.Bundles[Bundle_Indices]...)
    local Index = 1
    Re_Bundles = []
    for s in Bundle_Indices
        push!(Re_Bundles, Decomp_Zero.Bundles[s] .- (Decomp_Zero.Bundles[s][1] - Index))
        Index = Re_Bundles[end][end] + 1
    end
    Decomp = (
        Unreduced_Model = Decomp_Zero.Unreduced_Model[Select, :, Select],
        Data_Encoder = Decomp_Zero.Data_Encoder[Select, :, :],
        Data_Decoder = Decomp_Zero.Data_Decoder[:, :, Select],
        Bundles = Re_Bundles,
        Reduced_Model = Decomp_Zero.Reduced_Model[Select, :, Select],
        Reduced_Encoder = Decomp_Zero.Reduced_Encoder[Select, :, Select],
        Reduced_Decoder = Decomp_Zero.Reduced_Decoder[Select, :, Select],
    )
#     Decomp = InvariantModels.Select_Bundles(Decomp_Zero; How_Many=2, Time_Step = Time_Step, Ignore_Real = true)
#     Decomp = Select_Bundles_By_Energy(Index_List, Data, Encoded_Phase, Steady_State, SH, Decomp_Zero; How_Many=4, Time_Step=Time_Step)

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
        :Linear_Model    => Linear_Model,
        :Steady_State    => Steady_State,
        :SH              => SH,
        :Linear_Model_R  => Linear_Model_R,
        :Steady_State_R  => Steady_State_R,
        :SH_R            => SH_R,
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
    Selection = ([1;2],[3;4],[5;6]),
#     Selection = ([1;2],[3;4]),
    Model_Orders = (5, 3, 1),
    Encoder_Orders = (5, 3, 5),
#     Model_Orders = (7, 7),
#     Encoder_Orders = (7, 7),
    Unreduced_Model = Decomp.Unreduced_Model,
    Reduced_Model = Decomp.Reduced_Model,
    Reduced_Encoder = Decomp.Reduced_Encoder,
    SH = SH,
    Initial_Iterations = 16,
    Scaling_Parameter = 2^(-0),
    Initial_Scaling_Parameter = 2^(-0),
    Scaling_Order = Linear_Scaling,
    node_ratio = 0.8,
    leaf_ratio = 0.8,
    max_rank = 20,
    Linear_Type = (Encoder_Mean_Stiefel, Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
    Nonlinear_Type = (Encoder_Compressed_Full, Encoder_Compressed_Full, Encoder_Compressed_Local),
#     Linear_Type = (Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
#     Nonlinear_Type = (Encoder_Compressed_Full, Encoder_Compressed_Full),
    Name = "MTF-$(Name)-$(VER)",
    Time_Step = Time_Step,
)
MTFP_Test = Multi_Foliation_Test_Problem(
    MTFP,
    Index_List_T,
    Data_Decomp_T,
    Encoded_Phase_T;
    Initial_Scaling_Parameter = 2^(-0),
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
