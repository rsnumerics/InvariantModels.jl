# A simple autonomous system

## Identifying the invariant foliations

This tutorial goes through the steps of
* How to produce training and test data from a differential equation
* How to process the data so that it is suitable for finding invariant foliations
* How to fit invariant foliations to data
* How to calculate invariant manifolds from differential equations and discrete-time maps
* How to extract invariant manifolds from the identified invariant foliations
* How to assess the quality of obtained mathematical model

The differential equations is
```math
\dot{\boldsymbol{x}} = \boldsymbol{T}^{-1} \boldsymbol{f}(\boldsymbol{T} \boldsymbol{x}),
```
where 
```math
\boldsymbol{f}(\boldsymbol{x}) = 
               \begin{pmatrix}
                   -\alpha  x_1+\left(x_1^2+x_2^2\right) \left(\beta  x_1+\delta  x_2\right)+x_2 \\
                   -\alpha x_2+\left(x_1^2+x_2^2\right) \left(\beta  x_2-\delta  x_1\right)-x_1 \\
                   -2 \alpha  x_3
                \end{pmatrix}.
```
The equation is selected to be nonlinear, but easy to investigate, so that the automatic test code runs quickly on slow hardware.
Challenging problems can be found in the `Examples` folder.

Importing packages
```@example ct
using LinearAlgebra
using Tullio
using JLSO
using Random
using StaticArrays
using InvariantModels
using CairoMakie
```

### Creating data by solving a differential equation

Setting up the differential equations.
```@example ct
ODE_Transformation = SMatrix{3,3}(0.180047, 0.914719, -0.361763, 0.337552, -0.402896, -0.850725, -0.923927, 0.0310566, -0.381306)

function Tutorial_VF!(dx, x, u, Parameters)
    Alpha = Parameters.Alpha
    Beta = Parameters.Beta
    Delta = Parameters.Delta
    
    z = ODE_Transformation * x
    dz = SVector(
        -Alpha*z[1] + z[2] + (Beta*z[1] + Delta*z[2])*(z[1]^2 + z[2]^2),
        -z[1] - Alpha*z[2] + (-Delta*z[1] + Beta*z[2])*(z[1]^2 + z[2]^2),
        -2*Alpha*z[3],
        )
    dx .= transpose(ODE_Transformation) * dz
    
    return dx
end

# no forcing
function Tutorial_Forcing!(u, Alpha, Parameters)
    return u
end

function Tutorial_Forcing_Matrix!(x, Parameters, t)
    return Rigid_Rotation_Matrix!(x, [0], 0, t)
end
```

Setting up parameters, time steps, etc
```@example ct
Name = "Tutorial"
Skew_Dimension = 1
Training_Trajectories = 1
Testing_Trajectories = 1
Trajectory_Length = 2400

Forcing_Grid = Fourier_Grid(Skew_Dimension)
Parameters = (Alpha = 0.005, Beta = 0.005, Delta = 0.1)
Time_Step = 0.5

IC_x_Train = [0; 0.8; 0.1;;]
IC_x_Test = [0.6; 0.1; 0;;]
IC_Alpha_Train = ones(Skew_Dimension, Training_Trajectories)
IC_Alpha_Test = ones(Skew_Dimension, Testing_Trajectories)
IC_Force = []
nothing; #hide
```

Generating training and testing data.
Documentation is at [`Generate_From_ODE`](@ref).
```@example ct
List_of_Data, List_of_Phases = Generate_From_ODE(
    Tutorial_VF!,
    Tutorial_Forcing!,
    Tutorial_Forcing_Matrix!,
    Parameters,
    Time_Step,
    IC_x_Train,
    IC_Force,
    IC_Alpha_Train,
    ones(Int, Training_Trajectories) * Trajectory_Length,
)
List_of_Data_T, List_of_Phases_T = Generate_From_ODE(
    Tutorial_VF!,
    Tutorial_Forcing!,
    Tutorial_Forcing_Matrix!,
    Parameters,
    Time_Step,
    IC_x_Test,
    IC_Force,
    IC_Alpha_Test,
    ones(Int, Testing_Trajectories) * Trajectory_Length,
)
nothing; #hide
```

### Processing the data

Chopping up the trajectories into 600 point long segments.
There is a balance between short and long trajectory segments. 
Longer segments increase accuracy, but tracking error over longer periods can make the calculation unstable.
Documentation is at [`Chop_And_Stitch`](@ref).
```@example ct
Index_List, Data, Encoded_Phase =
    Chop_And_Stitch(List_of_Data, List_of_Phases; maxlen = 600)
Index_List_T, Data_T, Encoded_Phase_T =
    Chop_And_Stitch(List_of_Data_T, List_of_Phases_T; maxlen = 600)
nothing; #hide
```
    
A linear model is identified from the data. 
This model contains the steady state, the linear dynamics about the steady state,
and the forcing dynamics `SH`. Documentation is at [`Estimate_Linear_Model`](@ref).
```@example ct
Scaling = ones(size(Data, 2)) # 1 ./ ((2^-12) .+ sqrt.(sum(Data .^ 2, dims = 1)))
Steady_State, Linear_Model, SH = Estimate_Linear_Model(
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Iterations = 0,
    Order = 1,
)
nothing; #hide
```

If the model is forced it can be a good idea to filter the linear model so that it does not contain the high frequency content of the noise.
The foloowing line has no effect for autonomous systems, only included for completeness. Documentation is at [`Filter_Linear_Model`](@ref).
```@example ct
# filtering up to 2 harmonics
Linear_Model_Filtered = Filter_Linear_Model(Linear_Model, Forcing_Grid, 1)
nothing; #hide
```

Calculating the invariant vector bundles of the linear model, using the eigenvalues and eigenvectors of the transfer operator, see [`Create_Linear_Decomposition`](@ref). 
This creates tranformations that will bring the data into a coordinate system where the linear model is approximately block diagonal.
```@example ct
Decomp = Create_Linear_Decomposition(
    Linear_Model_Filtered,
    SH;
    Time_Step = Time_Step,
    Reduce = true,
    Align = true,
    By_Eigen = true,
)
nothing; #hide
```

Decomposing the data into the coordinate system of the invariant vector bundles. Documentation is at [`Decompose_Data`](@ref).
```@example ct
Data_Decomp, _ =
    Decompose_Data(Index_List, Data, Encoded_Phase, Steady_State, SH, Decomp.Data_Encoder)
Data_Decomp_T, _ = 
    Decompose_Data(Index_List_T, Data_T, Encoded_Phase_T, Steady_State, SH, Decomp.Data_Encoder)
nothing; #hide
```

Creating a scaling operation that makes sure that all data points are withing the unit ball of the phase space and that the signal amplitudes for each vector bundle are balanced. Documentation is at [`Decomposed_Data_Scaling`](@ref).
```@example ct
Data_Scale = Decomposed_Data_Scaling(Data_Decomp, Decomp.Bundles)
Data_Decomp .*= Data_Scale
Data_Decomp_T .*= Data_Scale
nothing; #hide
```

Saving the data, so that it can be used later on, when evaluating the accuracy of the identified invariant foliations.
```@example ct
JLSO.save(
    "DATA-$(Name).bson",
    :Parameters => Parameters,
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
nothing; #hide
```

### Fitting the data to a set of invariant foliations

Setting up the data structures of the invariant foliation. Here we select to hyper-parameters of the functional representations of the encoders and conjugate maps. 
We also select how the error should be scaled within the loss function as a function of signal amplitude.
Documentation is at [`Multi_Foliation_Problem`](@ref).
```@example ct
MTFP = Multi_Foliation_Problem(
    Index_List,
    Data_Decomp,
    Encoded_Phase,
    Selection = ([1; 2], [3;]),
    Model_Orders = (3, 1),
    Encoder_Orders = (1, 1),
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
    Nonlinear_Type = (Encoder_Dense_Latent_Linear, Encoder_Dense_Latent_Linear),
    Name = "MTF-$(Name)",
    Time_Step = Time_Step,
)
nothing; #hide
```

Setting up the data structures for the testing data. 
This is necessary, because `MTFP` includes estimates of initial conditions for latent trajectories, which are differentr for the testing data.
The estimates of initial conditions for latent trajectories must be updated before calculating the testing error. 
Documentation is at [`Multi_Foliation_Test_Problem`](@ref).
```@example ct
MTFP_Test = Multi_Foliation_Test_Problem(
    MTFP,
    Index_List_T,
    Data_Decomp_T,
    Encoded_Phase_T;
    Initial_Scaling_Parameter = 2^(-2),
)
nothing; #hide
```

Finally, we are fitting the invariant foliations to data.
Documentation is at [`Optimise!`](@ref).
```@example ct
Optimise!(
    MTFP,
    MTFP_Test;
    Model_Iterations = 16,
    Encoder_Iterations = 8,
    Steps = 12,
    Gradient_Ratio = 2^(-7),
    Gradient_Stop = 2^(-29),
)
nothing; #hide
```

## Analysing the the calculated invariant foliations


Setting the index of the vector bundle for which the results are analysed
```@example ct
Index = 1
```

Loading the data. Here it is not strictly necessary, because these are already in memory. 
However, if a separate script is use to analyse the results, data and the results must be loaded.
```@example ct
data = JLSO.load("DATA-$(Name).bson")
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
State_Dimension   = size(Data_Decomp, 1)
Skew_Dimension    = size(Encoded_Phase, 1)
IC_Force = []
nothing; #hide
```

Loading the identified invariant foliations.
```@example ct
dd = JLSO.load("MTF-$(Name).bson")
MTF = dd[:MTF]
XTF = dd[:XTF]
MTF_Test = dd[:Test_MTF]
XTF_Test = dd[:Test_XTF]
Error_Trace = dd[:Train_Error_Trace]
Test_Trace = dd[:Test_Error_Trace]
nothing; #hide
```

Creating a figure for plotting the results
```@example ct
fig = Create_Plot()
nothing; #hide
```

### Calculating the invariant manifold from the differential equation

Creating a polynomial vector field to be analysed subsequently.
Documentation is at [`Model_From_Function_Alpha`](@ref).
```@example ct
MM, MX, MD = Model_From_Function_Alpha(
    Tutorial_VF!,
    Tutorial_Forcing!,
    p -> Rigid_Rotation_Generator([0], 0.0),
    IC_Force,
    Parameters;
    State_Dimension = State_Dimension,
    Start_Order = 0,
    End_Order = 3,
)
nothing; #hide
```

Setting the parameters of the invariant manifold representation.
We use piecewise cubic polynomials in the radia direction and 11 Fourier collocation points in the angular direction.
The maximum amplutude to calculate is `Radius`.
```@example ct
Radius = 1.0
Radial_Order = 2
Radial_Intervals = 96
Polar_Order = 11
```

Calculating the invariant manifold from the vector field. 
Documentation is at [`Find_ODE_Manifold`](@ref).
```@example ct
MP, XP = Find_ODE_Manifold(
    MM, MX, MD,
    [1;2];
    Radial_Order = Radial_Order,
    Radial_Intervals = Radial_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 32,
    initial_maxiters = 200,
)
```

Plotting the backbone curves calculated from the vector field.
Documentation is at [`Plot_Model_Result!`](@ref).
```@example ct
    Plot_Model_Result!(fig, MP, XP, Hz = false)
```

### Calculating the invariant manifold from the discrete-time map

Creates a discrete-time model from the vector field. 
This is done by Taylor expanding a differential equation solver using automatic differentiation.
Documentation is at [`Model_From_ODE`](@ref).
```@example ct
MM, MX = Model_From_ODE(
    Tutorial_VF!,
    Tutorial_Forcing!,
    Tutorial_Forcing_Matrix!,
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
nothing; #hide
```

Setting the parameters of the invariant manifold representation.
```@example ct
Radius = 1.0
Cheb_Order = 2
Cheb_Intervals = 96
Polar_Order = 11
```

Calculating the invariant manifold from the discrete-time map.
Documentation is at [`Find_MAP_Manifold`](@ref).
```@example ct
PM, PX = Find_MAP_Manifold(
    MM, MX,
    [1;2];
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius,
    Phase_Dimension = Polar_Order,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 32,
    initial_maxiters = 200,
)
```

Plotting the backbone curves calculated from the discrete-time map.
Documentation is at [`Plot_Model_Result!`](@ref).
```@example ct
Plot_Model_Result!(
    fig,
    PM,
    PX,
    Time_Step = Time_Step,
    Hz = false,
    Damping_By_Derivative = true,
)
```

### Calculating the invariant manifold from the set of invariant foliations

Setting the parameters of the invariant manifold representation.
```@example ct
Radius = 1.0
Cheb_Order = 2
Cheb_Intervals = 112
Polar_Order = 17
```

Numerically calculating the invariant manifold from the identified invariant foliations.
At the same time a normal form of the conjugate dynamics is calculated numerically.
Documentation is at [`Find_DATA_Manifold`](@ref).
```@example ct
PPM, PPX = Find_DATA_Manifold(
    MTF,
    XTF,
    SH,
    Index;
    Radial_Order = Cheb_Order,
    Radial_Intervals = Cheb_Intervals,
    Radius = Radius,
    Phase_Dimension = Polar_Order,
    Transformation = Data_Decoder ./ reshape(Data_Scale, 1, 1, :),
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 36,
    initial_maxiters = 200,
)
```

Yet again, setting the parameters of the invariant manifold representation.
```@example ct
Radius = 1.0
Cheb_Order = 2
Cheb_Intervals = 120
Polar_Order = 17
```

Numerically calculating the invariant manifold from the identified invariant foliations. 
This has the same parametrisation as the invariant foliation and therefore can be used to reconstruct the invariant manifold from the encoded trajectories.
Documentation is at [`Extract_Manifold_Embedding`](@ref).
```@example ct
MIP, XIP, Torus, E_WW_Full, Latent_Data, E_ENC, AA, Valid_Ind = Extract_Manifold_Embedding(
    MTF, XTF, Index,
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
```

Plotting the backbone curves calculated from the invariant foliations.
Documentation is at [`Plot_Data_Result!`](@ref).
```@example ct
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
nothing; #hide
```

Plotting the training and testing error as they vary with amplitude.
Documentation is at [`Plot_Data_Error!`](@ref).
```@example ct
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
nothing; #hide
```

Plotting the history of training and testing error values for each iteration of the optimisation method.
Documentation is at [`Plot_Error_Trace`](@ref).
```@example ct
Plot_Error_Trace(fig, Index, Error_Trace, Test_Trace)
```

Annotate the final figure with necessary information.
```@example ct
Annotate_Plot!(fig)
save("FIGURE-FULL.svg", fig) #hide
nothing; #hide
```
![](FIGURE-FULL.svg)

### Evaluating the model and comparing to testing data

Plotting the encoded testing trajectory and the predicted testing trajectory in the latent space. 
The green line is the difference between prediction and the unseen data.
```@example ct
fig2 = Figure()

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
save("FIGURE-LATENT.svg", fig2) #hide
nothing; #hide
```

![](FIGURE-LATENT.svg)
