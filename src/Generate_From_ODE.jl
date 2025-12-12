# SPDX-License-Identifier: EUPL-1.2

# generates
# Output: Index_List, Data, Encoded_Phase, List_Of_Trajectories, List_Of_Phases
#   - Index_List
#       Contains the start index minus one of each trajectory. The last index is the number of all data points
#       To access trajectory 'k', we use
#           Data[:, Index_List[k]+1:Index_List[k+1]]
#   - Data
#       The state space samples every Time_Step it is a single matrix
#   - Encoded_Phase
#       Value of the phase variable encoded on a 2*Fourier_Order + 1 dimensional linear space
#   - List_Of_Trajectories
#       A list of matrices, each containing a trajectory
#   - List_Of_Phases
#       the not yet encoded phase variable

function Power_Scaling(p, delta, len)
    if len > 1
        alpha = delta^(p - 1)
        pt = range(0, 1, length = len + 1)
        thr = floor(delta * len)
        sc1 = alpha * (1:thr) / len
        sc2 = (((thr + 1):len) / len) .^ p
        return vcat(sc1, sc2)
    else
        return [1.0]
    end
end

# Vectorfield!(x, y, Parameters, Alpha)
# Alpha_Map!(Alpha, Parameters, IC_Alpha, t)
# Usage
#   L, P = Generate_From_ODE(shawpierre!, (x,y,t) -> Rigid_Rotation!(x, Forcing_Grid, Omega_ODE, y, t), Parameters, Time_Step, IC_x, IC_Alpha, [1200])

"""
    Generate_From_ODE(
        Vectorfield!,
        Forcing_Map!,
        Alpha_Map!,
        Parameters,
        Time_Step,
        IC_x,
        IC_Force,
        IC_Alpha,
        Trajectory_Lengths,
    )

Generates a data set from am ordinary differential equation (ODE). The ODE is defined by the three functions
`Vectorfield!`, `Forcing_Map!` and `Alpha_Map!`.
* `Time_Step` is the sampling time interval of the soluation
* `IC_x` each column contains an initial condition in the state space
* `IC_Force` this is a storage space and will be overwritten with the actual value of the forcing state `Alpha`
* `IC_Alpha` each column contains an initial condition of the forcing state
* `Trajectory_Lengths` a vector of trajectory lengths

The return values are `List_Of_Trajectories`, `List_Of_Phases`, which can be further processed by identifying a linear model
[`Estimate_Linear_Model`](@ref).

An example of the ODE is
```code
function Vectorfield!(dx, x, u, Parameters)
    ... dx is the dx/dt
    ... x is the state variable vector
    ... u is the time varying input or forcing vector
    ... Parameters is a data structure that is passed around all function
        and contains all auxilliary information necessary to calculate dx
    ...
    return x
end

function `Forcing_Map!`(u, Alpha, Parameters)
    ... This function produces the forcing vector 'u'
    ... The state variable of the forcing is `Alpha`
    ... Parameters is a data structure that is passed around all function
    ... The forcing depends on the state variable of the forcing `Alpha`
    ... In this case the forcing is just a linear combination of `Alpha`
    u[1:2] .= Parameters.Weights * Alpha
    return u
end

function Alpha_Map!(x, Parameters, t)
    ... Returns a matrix, which is the fundamental solution of the forcing dynamics
    ... 'x' is the fundamental solution
    ... Parameters is a data structure that is passed around all function
    ... `t` is the independent time variable
    return Rigid_Rotation_Matrix!(x, Parameters.Forcing_Grid, Parameters.Omega, t)
end
```
"""
function Generate_From_ODE(
        Vectorfield!,
        Forcing_Map!,
        Alpha_Map!,
        Parameters,
        Time_Step,
        IC_x,
        IC_Force,
        IC_Alpha,
        Trajectory_Lengths,
    )
    # the output
    List_Of_Trajectories = Array{Array{Float64}, 1}(undef, length(Trajectory_Lengths))
    List_Of_Phases = Array{Array{Float64}, 1}(undef, length(Trajectory_Lengths))
    # local variable
    Alpha_Matrix = zeros(eltype(IC_Alpha), size(IC_Alpha, 1), size(IC_Alpha, 1))
    for k in eachindex(Trajectory_Lengths)
        Sample_Times = range(0, step = Time_Step, length = Trajectory_Lengths[k])
        ODE_Problem = ODEProblem(
            (x, y, p, t) -> Vectorfield!(
                x,
                y,
                Forcing_Map!(IC_Force, Alpha_Map!(Alpha_Matrix, p, t) * IC_Alpha[:, k], p),
                p,
            ),
            IC_x[:, k],
            (Sample_Times[1], Sample_Times[end]),
            Parameters,
        )
        Solution = solve(
            ODE_Problem,
            Feagin14(),
            abstol = 2 * eps(Float64),
            reltol = 2 * eps(Float64),
            tstops = Sample_Times,
        )
        List_Of_Trajectories[k] = reduce(hcat, Solution(Sample_Times).u)
        Phases = zeros(eltype(Alpha_Matrix), size(IC_Alpha, 1), Trajectory_Lengths[k])
        for j in eachindex(Sample_Times)
            Phases[:, j] .=
                Alpha_Map!(Alpha_Matrix, Parameters, Sample_Times[j]) * IC_Alpha[:, k]
        end
        List_Of_Phases[k] = Phases
    end
    return List_Of_Trajectories, List_Of_Phases
end

# Vectorfield! is (x, y, p, t1, t2) -> nothing
#   a mutating function
#       x: \dot{y}
#       y: state of system
#       p: parameters of the system
#       t1: [0, 2pi) the phase within a sampling time-step
#       t2: Omega * t, the phase of external forcing

function Generate_Data_From_ODE(
        State_Dimension,
        Vectorfield!,
        Parameters,
        Maximum_Initial_Condition,
        Number_Of_Trajectories,
        Trajectory_Length,
        Fourier_Order,
        Time_Step,
        Omega;
        Transient_Steps = 4000,
        Steady_Steps = 100,
        IC_Matrix = [],
    )
    Initial_Condition = zeros(State_Dimension)
    #     Grid = getgrid(Fourier_Order)
    # hope it converges...
    Period = 2 * pi / Omega

    # creating QP solution
    Time_Span = (0, Transient_Steps * Time_Step)
    ODE_Problem = ODEProblem(
        (x, y, p, t) ->
        Vectorfield!(x, y, p, 2 * pi / Time_Step * (t - Time_Span[1]), Omega * t),
        Initial_Condition,
        Time_Span,
        Parameters,
    )
    Solution = solve(ODE_Problem, Vern7(), abstol = 1.0e-8, reltol = 1.0e-8)
    Time_End = Time_Step * floor(Time_Span[end] / Time_Step)
    Steady_State_Sample_Times =
        range(Time_End - (Steady_Steps - 1) * Time_Step, Time_End, step = Time_Step)
    Steady_State_Samples = [Solution(t) for t in Steady_State_Sample_Times]
    Steady_State_Phases = [mod(Omega * t, 2 * pi) for t in Steady_State_Sample_Times]
    Steady_State_Order = sortperm(Steady_State_Phases)
    Steady_State = Interpolations.extrapolate(
        interpolate(
            (Steady_State_Phases[Steady_State_Order],),
            Steady_State_Samples[Steady_State_Order],
            Gridded(Linear()),
        ),
        Periodic(),
    )
    # Plotting
    Steady_State_Matrix = zeros(State_Dimension, length(Steady_State_Order))
    for (k, p) in enumerate(Steady_State_Order)
        Steady_State_Matrix[:, k] = Steady_State_Samples[p]
    end
    pl = lineplot(Steady_State_Phases[Steady_State_Order], Steady_State_Matrix')
    display(pl)
    # End plotting
    function Out_Of_Bounds_Condition(u, t, Integrator)
        return norm(u) > 200 * maximum(Maximum_Initial_Condition)
    end
    function Stop_Simulation!(Integrator)
        println("Out of bounds solution, terminating.")
        return terminate!(Integrator)
    end
    #
    List_Of_Trajectories = Array{Array{Float64}, 1}(undef, Number_Of_Trajectories)
    List_Of_Phases = Array{Array{Float64}, 1}(undef, Number_Of_Trajectories)
    # setting initial conditions
    Initial_Condition_Matrix = zeros(State_Dimension, Number_Of_Trajectories)
    if isempty(IC_Matrix)
        Gaussian_Initial_Conditions = randn(State_Dimension, Number_Of_Trajectories)
        Initial_Condition_Matrix .=
            Gaussian_Initial_Conditions ./
            sqrt.(sum(Gaussian_Initial_Conditions .^ 2, dims = 1)) .* reshape(
            Power_Scaling(2 / State_Dimension, 0.1, Number_Of_Trajectories),
            1,
            Number_Of_Trajectories,
        )
        Initial_Condition_Matrix .*= reshape(Maximum_Initial_Condition, :, 1)
    else
        Initial_Condition_Matrix .= IC_Matrix
    end
    Skew_Dimension = 2 * Fourier_Order + 1
    Grid = Fourier_Grid(Skew_Dimension)
    t0s = Grid[rand(1:Skew_Dimension, Number_Of_Trajectories)] ./ (2 * pi)
    for j in 1:Number_Of_Trajectories
        Phase_Shift = Period * t0s[j]
        Initial_Condition .= Initial_Condition_Matrix[:, j]
        Initial_Condition .+= Steady_State(mod(Phase_Shift * Omega, 2 * pi))

        Time_Span = (Phase_Shift, Time_Step * (Trajectory_Length + 1) + Phase_Shift) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
        ODE_Problem = ODEProblem(
            (x, y, p, t) -> Vectorfield!(
                x,
                y,
                p,
                2 * pi / Time_Step * (t - Time_Span[1]),
                Omega * t,
            ),
            Initial_Condition,
            Time_Span,
            Parameters,
            callback = DiscreteCallback(Out_Of_Bounds_Condition, Stop_Simulation!),
        )
        Sample_Times = range(start = Time_Span[1], stop = Time_Span[2], step = Time_Step)
        Solution = solve(
            ODE_Problem,
            Feagin14(),
            abstol = 2 * eps(Float64),
            reltol = 2 * eps(Float64),
            tstops = Sample_Times,
        )
        #             @show length(1+(j-1)*Trajectory_Length:j*Trajectory_Length), size([sum(Solution(t))/length(Solution(t)) for t in Sample_Times])
        List_Of_Trajectories[j] = reduce(hcat, Solution(Sample_Times).u)
        List_Of_Phases[j] = collect(Sample_Times * Omega)
    end
    # Compose trajectories into arrays
    Length_List = [size(x, 2) for x in List_Of_Trajectories]
    Index_List = vcat(0, cumsum(Length_List))
    Data = hcat(List_Of_Trajectories...)
    List_Of_Encoded_Phase = [
        Fourier_Interpolate(Fourier_Grid_Of_Order(Fourier_Order), ph) for
            ph in List_Of_Phases
    ]
    Encoded_Phase = hcat(List_Of_Encoded_Phase...)
    #
    return Index_List, Data, Encoded_Phase, List_Of_Trajectories, List_Of_Encoded_Phase
end
