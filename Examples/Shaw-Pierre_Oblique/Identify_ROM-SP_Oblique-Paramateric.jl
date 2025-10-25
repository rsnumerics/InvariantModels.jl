Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module ct
using LinearAlgebra
using Tullio
using JLSO
using Distributions
using Random
using Main.InvariantModels

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

# the parameter is approximated by orthogonal collocation
function Shaw_Pierre_Forcing!(u, Alpha, Parameters)
    u[1:2] .= 0
    u[3] = dot(Parameters.Forcing_Grid, Alpha) # the identity function
    return u
end

# the parameter does not change over time
function Shaw_Pierre_Forcing_Matrix!(x, Parameters, t)
    return I
end

Name = "SP_Oblique"
VER = "Parametric"
DATAVER = "Parametric"
Generate = true
Process = true

# parameter dependant
Skew_Dimension = 5
Training_Trajectories = 16
Testing_Trajectories = 1
Forcing_Amplitude = 0.0
Omega_ODE = 2^(1/4)
Trajectory_Length = 1350

# it is a Chebyshev collocation
Forcing_Grid = Chebyshev_Grid(Skew_Dimension, 0.4, 0.6)
Parameters = (
    Forcing_Grid = Forcing_Grid,
    Amplitude = Forcing_Amplitude,
    Omega = Omega_ODE,
)
Time_Step = 2 * pi / 19 / Omega_ODE

Max_Amplitude = 0.6
if Generate
    AA = randn(4, Training_Trajectories)
    IC_x_Train =
        Max_Amplitude *
        AA *
        diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
        diagm(0.95 .+ 0.1 .* rand(size(AA, 2)))
    AA = randn(4, Testing_Trajectories)
    IC_x_Test =
        0.95 *
        Max_Amplitude *
        AA *
        diagm(1 ./ sqrt.(vec(sum(AA .^ 2, dims = 1)))) *
        diagm(0.95 .+ 0.1 .* rand(size(AA, 2)))

    # not random, but uniformly distributed
    function Random_Phase(Skew_Dimension, Trajectories)
        if Trajectories > 1
            IC_Alpha = transpose(Barycentric_Interpolation_Matrix(Forcing_Grid, Chebyshev_Grid(Trajectories, Forcing_Grid[1], Forcing_Grid[end])))
        else
            IC_Alpha = transpose(Barycentric_Interpolation_Matrix(Forcing_Grid, rand(Uniform(Forcing_Grid[1], Forcing_Grid[end]), Trajectories)))
        end
        return IC_Alpha
    end

    IC_Alpha_Train = Random_Phase(Skew_Dimension, Training_Trajectories)
    IC_Alpha_Test = Random_Phase(Skew_Dimension, Testing_Trajectories)

    IC_Force = [0.0; 0.0; 0.5]
    List_of_Data, List_of_Phases = Generate_From_ODE(
        Shaw_Pierre_VF!,
        Shaw_Pierre_Forcing!,
        Shaw_Pierre_Forcing_Matrix!,
        Parameters,
        Time_Step,
        IC_x_Train,
        IC_Force,
        IC_Alpha_Train,
        ones(Int, Training_Trajectories) * Trajectory_Length,
    )
    List_of_Data_T, List_of_Phases_T = Generate_From_ODE(
        Shaw_Pierre_VF!,
        Shaw_Pierre_Forcing!,
        Shaw_Pierre_Forcing_Matrix!,
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
        Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 449)
    Index_List_T, Data_T, Encoded_Phase_T =
        Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 449)

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
    # DO NOT FILTER
    # DO NOT REDUCE
    # filtering up to 2 harmonics
    #     Linear_Model_Filtered = Filter_Linear_Model(Linear_Model, Forcing_Grid, 1)
    Decomp = Create_Linear_Decomposition(
        Linear_Model,
        SH;
        Time_Step=Time_Step,
        Reduce=false,
        Align=true,
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
        :Linear_Model    => Linear_Model,
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
    Selection = ([1; 2], [3; 4]),
    Model_Orders = (3, 1),
    Encoder_Orders = (3, 1),
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
    Linear_Type = (Encoder_Array_Stiefel, Encoder_Array_Stiefel),
    Nonlinear_Type = (Encoder_Dense_Full, Encoder_Dense_Full),
#     Linear_Type = (Encoder_Mean_Stiefel, Encoder_Mean_Stiefel),
#     Nonlinear_Type = (Encoder_Dense_Full, Encoder_Dense_Full),
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

