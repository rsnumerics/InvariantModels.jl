# PROCEDURE
# 1. Cut from of the signal
# 2. Cut tail of the signal
# 3. Delay Embed
# 4. Chop up trajectories into manageable length
# 5. Pich out: Training data, Testing data

# for plotting frequency response
# function FIRfreqz(b::Array, w = range(0, stop=π, length=1024))
#     n = length(w)
#     h = Array{eltype(b)}(undef, n)
#     sw = 0
#     for i = 1:n
#         for j = 1:length(b)
#             sw += b[j]*exp(-im*w[i])^-j
#         end
#         h[i] = sw
#         sw = 0
#     end
#     return h
# end
#
# function Bandpass_Filter(Signal, fs, low, high; Threshold=10.0)
#     wsize = 2 ^ ceil(Int, log2(2 * fs / low))
#     println("Window size=", wsize, " fs=", fs, " low=", low, " high=", high)
#     Kernel = DSP.digitalfilter(DSP.Bandpass(2*low/fs, 2*high/fs), DSP.FIRWindow(DSP.Windows.hanning(wsize-1)))
#     Result = zero(Signal)
#     for k in axes(Signal, 1)
#         view(Result, k,:) .= DSP.filtfilt(Kernel, Signal[k, :])
#     end
#     return Result
# #     return Result[:, 1:size(Signal, 2)]
# end
#
# function Highpass_Filter(Signal, fs, low)
#     wsize = 2 ^ ceil(Int, log2(2 * fs / low))
#     println("Window size=", wsize, " fs=", fs, " low=", low)
#     Kernel = DSP.digitalfilter(DSP.Highpass(2*low/fs), DSP.FIRWindow(DSP.Windows.hanning(wsize-1)))
#     Result = zero(Signal)
#     for k in axes(Signal, 1)
#         view(Result, k,:) .= DSP.filtfilt(Kernel, Signal[k, :])
#     end
#     return Result
# end

# STEP 2
# Takes a moving average (window)
# and cuts off the tail of each signal
# that is less that (ratio) times the amplitude
# of the average at the last window
function Cut_Signals(Signals; window = 60, ratio = 2)
    moving_average(vs, n) =
        [sum(view(vs, :, i:(i+n-1)), dims = 2) / n for i = 1:(size(vs, 2)-(n-1))]
    function amplitude(vs, mavg, n)
        [
            sqrt.(maximum(sum((view(vs, :, i:(i+n-1)) .- mavg[i]) .^ 2, dims = 1))) for
            i = 1:(size(vs, 2)-(n-1))
        ]
    end
    cut = []
    for sig in Signals
        mavg = moving_average(sig, window)
        amp = amplitude(sig, mavg, window)
        push!(cut, findlast(amp .> ratio * amp[end]))
    end
    Trimmed = [Signals[k][:, 1:cut[k]] for k in eachindex(Signals)]
    return Trimmed
end

# STEP 3
#
function Delay_Embed_Proper(Signals, Phases; delay = 1, skip = 1, delay_step = skip)
    State_Dimension = size(Signals[1], 1)
    Factor = div(delay, delay_step)
    Delay_State_Dimension = (1 + Factor) * State_Dimension
    Delay_Signals = []
    Delay_Phases = []
    for (traj, phase) in zip(Signals, Phases)
        Points = 1 + div(size(traj, 2) - 1 - Factor * delay_step, skip)
        D_Traj = zeros(eltype(traj), Delay_State_Dimension, Points)
        for k = 0:Factor
            #             @show size(D_Traj[1+k*State_Dimension:(k+1)*State_Dimension, :]) , size(traj[:, range(1 + k * delay_step, step=skip, length=Points)])
            D_Traj[(1+k*State_Dimension):((k+1)*State_Dimension), :] .=
                traj[:, range(1 + k * delay_step, step = skip, length = Points)]
        end
        D_Phase = zeros(eltype(phase), size(phase, 1), Points)
        D_Phase .= phase[:, range(1, step = skip, length = Points)]
        push!(Delay_Signals, D_Traj)
        push!(Delay_Phases, D_Phase)
    end
    return Delay_Signals, Delay_Phases
end
#
@doc raw"""
    Delay_Embed(Signals, Phases; delay = 1, skip = 1, max_length = typemax(Int))

Creates a delay embedding of `Signals` while limits trajectory length to `max_length`.
The delay is `delay` long and only every `skip`th point along the trajectory is put into the delay embedded data.
`Phases` are simply copied over to the output while making sure that signal lengths and phase length are the same, limited by `max_length`.

The return values `Delay_Signals`, `Delay_Phases` are matrices of appropriate sizes.
"""
function Delay_Embed(Signals, Phases; delay = 1, skip = 1, max_length = typemax(Int))
    State_Dimension = size(Signals[1], 1)
    Factor = div(delay, skip)
    Delay_State_Dimension = Factor * State_Dimension
    Delay_Signals = []
    Delay_Phases = []
    for (traj, phase) in zip(Signals, Phases)
        R_range = range(0, min(size(traj, 2), max_length) - Factor, step = skip)
        D_Traj = zeros(eltype(traj), Delay_State_Dimension, length(R_range))
        D_Phase = zeros(eltype(phase), size(phase, 1), length(R_range))
        for k = 1:Factor
            D_Traj[(1+(k-1)*State_Dimension):(k*State_Dimension), :] .= traj[:, k.+R_range]
        end
        D_Phase .= phase[:, 1 .+ R_range]
        push!(Delay_Signals, D_Traj)
        push!(Delay_Phases, D_Phase)
    end
    return Delay_Signals, Delay_Phases
end

# STEP 4
function Chop_Signals(Signals, Phases; maxlen = 1000)
    Chopped_Signals = []
    Chopped_Phases = []
    for (traj, phase) in zip(Signals, Phases)
        chunk_size = div(size(traj, 2), div(size(traj, 2), maxlen) + 1) - 1
        @show chunk_size, size(traj, 2), maxlen
        S_range = range(1, size(traj, 2), step = chunk_size)
        @show length(S_range), S_range, size(traj, 2)
        for k = 1:(length(S_range)-1)
            push!(Chopped_Signals, view(traj, :, S_range[k]:S_range[k+1]))
            push!(Chopped_Phases, view(phase, :, S_range[k]:S_range[k+1]))
        end
    end
    return Chopped_Signals, Chopped_Phases
end

@doc raw"""
    Chop_And_Stitch(Signals, Phases; maxlen = 1000)

Chops up `Signals` and `Phases` into chunks of maximum `maxlen` chuncks.
Then concatenates the result into the arrays
`Index_List`, `Data`, `Encoded_Phase`.
The list `Index_List` tells, where each chunk starts and ends within the arrays `Data` and `Encoded_Phase`.
"""
function Chop_And_Stitch(Signals, Phases; maxlen = 1000)
    Chopped_Signals, Chopped_Phases = Chop_Signals(Signals, Phases; maxlen = maxlen)
    Data = hcat(Chopped_Signals...)
    Encoded_Phase = hcat(Chopped_Phases...)
    Index_List = [0; cumsum([size(traj, 2) for traj in Chopped_Signals])]
    return Index_List, Data, Encoded_Phase
end

function Bin_Cuts(Amplitudes, max_points, nbins)
    amp_max = maximum(Amplitudes)
    amp_min = minimum(Amplitudes)
    # bounds contains the bin boundaries (starting with zero)
    bounds = range(amp_min, amp_max, length = nbins + 1)
    # bins will contain the indices for the amplitudes
    bins = Array{Array{Int,1},1}(undef, nbins)
    for k in eachindex(bins)
        bins[k] = findall((Amplitudes .>= bounds[k]) .&& (Amplitudes .< bounds[k+1]))
    end
    # ll has the lengths of each bin
    ll = length.(bins)
    llp = sortperm(ll, rev = true)
    # lls has the biggest bin first
    lls = ll[llp]
    # finding out home many element should be in each box
    let k = 1
        to_rem = sum(lls) - max_points
        @show to_rem
        while to_rem > 0 && k < nbins
            # how much can be removed just by chopping
            rem = sum(lls[1:k] .- lls[k+1])
            if rem <= to_rem && (k + 1) < nbins
                lls[1:k] .= lls[k+1]
                to_rem -= rem
            else
                drem = div(to_rem, k + 1)
                lls[1:(k+1)] .-= drem
                to_rem -= (k + 1) * drem
            end
            k += 1
        end
    end
    # putting back the relevant numbers
    ll = lls[invperm(llp)]
    for k in eachindex(bins)
        bp = randperm(length(bins[k]))
        bins[k] = bins[k][bp[1:ll[k]]]
    end
    indices = sort(vcat(bins...))
    return indices
end

function Pick_Signals(Signals, ratio_train, ratio_test, nbins = 30)
    amps = [mean(sqrt.(sum(traj .^ 2, dims = 1))) for traj in Signals]
    len_train = round(Integer, ratio_train * length(amps))
    len_test = round(Integer, ratio_test * length(amps))
    if len_train + len_test > length(amps)
        len_test = length(amps) - len_train
    end

    indices = Bin_Cuts(amps, len_train + len_test, nbins)
    perm = randperm(length(indices))
    train = sort(indices[perm[1:max(1, min(length(perm) - 1, len_train))]])
    test = sort(
        indices[perm[min(length(perm), len_train + 1):min(
            length(perm),
            len_train + len_test,
        )]],
    )

    return train, test
end

function Linear_Decompose(
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Time_Step = 1.0,
    Dimensions = size(Data, 1),
    Iterations = 0,
    Order = 1,
    Avoid_Real = false,
    Aut_Epsilon = 1e-3,
)
    State_Dimension = size(Data, 1)
    Skew_Dimension = size(Encoded_Phase, 1)
    Trajectories = length(Index_List) - 1
    #
    MM = MultiStep_Model(State_Dimension, Skew_Dimension, 0, Order, Trajectories)
    MX = zero(MM)
    From_Data!(MM, MX, Index_List, Data, Encoded_Phase, Scaling)
    # should not be the identity
    SH = deepcopy(MM.SH)
    if maximum(abs.(SH - I)) < Aut_Epsilon
        println("Linear_Decompose: Near autononomous system. Perturbing the shift matrix.")
        MM.SH .+= 2 * Aut_Epsilon * randn(size(MM.SH))
        F = svd(MM.SH)
        MM.SH .= F.U * F.Vt
    end
    if Iterations > 0
        MM_Cache = Make_Cache(MM, MX, Index_List, Data, Encoded_Phase, Scaling)
        #         Beta, LL, NM = Optimise!(MM, MX, Index_List, Data, Encoded_Phase, Scaling, Cache = MM_Cache)
        #         @show Beta, LL, NM
        optfun = Optimization.OptimizationFunction(
            (x, p) -> Loss_With_Update(
                MM,
                x,
                Index_List,
                Data,
                Encoded_Phase,
                Scaling,
                Cache = MM_Cache,
            ),
            grad = (g, x, p) -> Gradient!(
                g,
                MM,
                x,
                Index_List,
                Data,
                Encoded_Phase,
                Scaling,
                Cache = MM_Cache,
            ),
        )
        prob = Optimization.OptimizationProblem(optfun, MX, [])
        osol = Optimization.solve(
            prob,
            Optim.GradientDescent(alphaguess = 1 * eps(1.0));
            maxiters = Iterations,
        )
        display(osol)
        display(osol.stats)
        MX .= osol.u
    end
    #
    @time SS, BB_Linear = Find_Torus(MM, MX)
    #
    LVt, LW, Lambda, _, _, Lambda_Diagonal_C, Real_Index_Start =
        Decompose_Model_Right(BB_Linear, MM.SH; Time_Step = Time_Step)
    EV_Pre = [
        eigvals(Lambda[(1+2*(k-1)):(2*k), (1+2*(k-1)):(2*k)]) for
        k = 1:div(Real_Index_Start - 1, 2)
    ]
    EV = vcat(EV_Pre..., diag(Lambda[Real_Index_Start:end, Real_Index_Start:end]))
    #     EV[findall(abs.(EV) .> 1)] .= eps(1.0)
    Sorted = sortperm(abs.(EV), rev = true)
    #     println("Linear_Decompose: Lambda_Diagonal")
    #     display(EV[Sorted[1:min(Dimensions+1, length(Sorted))]])
    println("Linear_Decompose: Lambda_Diagonal [rad/s]")
    display(log.(EV[Sorted[1:min(Dimensions + 1, length(Sorted))]]) ./ Time_Step)
    local Reduced_All
    if Dimensions < size(Data, 1)
        if isapprox(abs(EV[Sorted[Dimensions]]), abs(EV[Sorted[Dimensions+1]]))
            println("Linear_Decompose: adding a dimension.")
            Reduced_All = Sorted[1:(Dimensions+1)]
        else
            Reduced_All = Sorted[1:Dimensions]
        end
    else
        Reduced_All = Sorted[1:Dimensions]
    end
    #     Reduced_Complex = sort(Reduced_All[findall(Reduced_All .< Real_Index_Start)])
    #     Reduced_Real = sort(Reduced_All[findall(Reduced_All .>= Real_Index_Start)])
    Reduced_Complex = Reduced_All[findall(Reduced_All .< Real_Index_Start)]
    Reduced_Real = Reduced_All[findall(Reduced_All .>= Real_Index_Start)]
    if Avoid_Real
        Reduced_Indices = Reduced_Complex
    else
        Reduced_Indices = vcat(Reduced_Complex, Reduced_Real)
    end
    println("Linear_Decompose: Decay rates of the resolved modes [rad/s]")
    display(log.(EV[Reduced_Indices]) ./ Time_Step)
    println("Linear_Decompose: Decay rates of the resolved modes [Hz]")
    display(log.(EV[Reduced_Indices]) ./ (2 * pi * Time_Step))
    #
    LVt_Reduced = LVt[Reduced_Indices, :, :]
    #
    # keep the variation!
    # LVt .= 0.5 * LVt .+ 0.5 * mean(LVt, dims=3)
    Data_Var = Data - SS * Encoded_Phase
    @tullio Data_Full[i, k] := LVt[i, q, p] * Data_Var[p, k] * Encoded_Phase[q, k]
    magsort = sortperm(
        [sqrt(dot(Data_Full[k, :], Data_Full[k, :])) for k in axes(Data_Full, 1)],
        rev = true,
    )
    display(log.(EV[magsort]) ./ (2 * pi * Time_Step))

    @tullio Data_Decomposed[i, k] :=
        LVt_Reduced[i, q, p] * Data_Var[p, k] * Encoded_Phase[q, k]
    return Data_Decomposed,
    EV[Reduced_Indices],
    SS,
    LVt_Reduced,
    LW[:, :, Reduced_Indices],
    BB_Linear,
    SH
end

@doc raw"""
    Filter_Linear_Model(BB, Grid, order)

Assuming a uniform Fourier grid `Grid`, this function returns a truncated Fourier series of the linear model `BB`.
This removes inaccurate higher frequency components of the approximate linear model before decomposing it into invariant vector bundles.
"""
function Filter_Linear_Model(BB, Grid, order)
    BB_Filtered = zero(BB)
    BB_Filtered .+= mean(BB, dims = 2)
    Max_Order = div(length(Grid) - 1, 2)
    for k = 1:min(order, Max_Order)
        BB_Filtered .+=
            sum(reshape(cos.(k * Grid) / (length(Grid) / 2), 1, :, 1) .* BB, dims = 2) .*
            reshape(cos.(k * Grid), 1, :, 1)
        BB_Filtered .+=
            sum(reshape(sin.(k * Grid) / (length(Grid) / 2), 1, :, 1) .* BB, dims = 2) .*
            reshape(sin.(k * Grid), 1, :, 1)
    end
    @show norm(BB - BB_Filtered)
    return BB_Filtered
end

@doc raw"""
    Estimate_Linear_Model(
        Index_List,
        Data,
        Encoded_Phase,
        Scaling;
        Iterations = 0,
        Order = 1,
    )

Given the data `Index_List`, `Data`, `Encoded_Phase` this function estimates a linear model, a steady state and a model of forcing.

The output is `Steady_State`, `BB_Linear`, `SH`. `Steady_State` is the steady state of the system,
`BB_Linear` is the estimated linear model and `SH` is the unitary transformation representing forcing.

When fitting the linear model `Scaling` attaches an importance to each data point.
`Iterations` allows us to take into account the data as trajectories and use multi-step optimisation to find `BB_Linear`.
A nonlinear model can also be fitted by making `Order` ``> 1``.
When a nonlinear model is identified,
its steady state is found by Newton's method and its Jacobian is calculated at the steady state,
which is then returned as the linear model `BB_Linear`.
"""
function Estimate_Linear_Model(
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Iterations = 0,
    Order = 1,
)
    State_Dimension = size(Data, 1)
    Skew_Dimension = size(Encoded_Phase, 1)
    Trajectories = length(Index_List) - 1
    #
    MM = MultiStep_Model(State_Dimension, Skew_Dimension, 0, Order, Trajectories)
    MX = zero(MM)
    From_Data!(MM, MX, Index_List, Data, Encoded_Phase, Scaling)
    if Iterations > 0
        MM_Cache = Make_Cache(MM, MX, Index_List, Data, Encoded_Phase, Scaling)
        Maximum_Radius = 4 * sqrt(manifold_dimension(MM))
        Trust_Radius = 1.0
        t0 = time()
        println("Estimate_Linear_Model: Starting Optimisation")
        for it in 1:Iterations
            Trust_Radius, M_Loss, M_Grad = Optimise!(
                MM, MX, Index_List, Data, Encoded_Phase, Scaling,
                Cache=MM_Cache, Radius=Trust_Radius, Maximum_Radius=Maximum_Radius
            )
            println(
                "    Step=$(it). " *
                @sprintf("time = %.1f[s] ", time() - t0) *
                @sprintf("F(x) = %.5e ", M_Loss) *
                @sprintf("G(x) = %.5e ", M_Grad) *
                @sprintf("R = %.5e ", Trust_Radius)
            )
            if Trust_Radius >= Maximum_Radius
                Trust_Radius = 1.0
                return break
            end
         end
#         optfun = Optimization.OptimizationFunction(
#             (x, p) -> Loss_With_Update(
#                 MM,
#                 x,
#                 Index_List,
#                 Data,
#                 Encoded_Phase,
#                 Scaling,
#                 Cache = MM_Cache,
#             ),
#             grad = (g, x, p) -> Gradient!(
#                 g,
#                 MM,
#                 x,
#                 Index_List,
#                 Data,
#                 Encoded_Phase,
#                 Scaling,
#                 Cache = MM_Cache,
#             ),
#         )
#         prob = Optimization.OptimizationProblem(optfun, MX, [])
#         osol = Optimization.solve(
#             prob,
#             Optim.GradientDescent(alphaguess = 1 * eps(1.0));
#             maxiters = Iterations,
#         )
#         display(osol)
#         display(osol.stats)
#         MX .= osol.u
    end
    #
    @time SS, BB_Linear = Find_Torus(MM, MX)
    return SS, BB_Linear, deepcopy(MM.SH)
end

@doc raw"""
    Create_Linear_Decomposition(
        BB_Filtered,
        SH;
        Time_Step = 1.0,
        Reduce = false,
        Align = true,
        By_Eigen = false,
    )

Creates an invariant vector bundle decomposition of the linear system represented by `BB_Filtered` and the forcing dynamics `SH`.
If `Time_Step` is specified, the fucntion also prints the estimated natural frequencies and damping ratios.
The decomposition outputs vector bundles with orthonormal bases.
Therefore the reduced model will be block-diagonal (with at most ``2\times 2`` blocks), but not autonomous.
If `Reduce == true` the system will be reduced to an autonomous form and the vector bundles will only be orthogonal in a weaker sense,
averaged over the phase space of the forcing dyamics.

The output is a named tuple
```
(
    Unreduced_Model,
    Data_Encoder,
    Data_Decoder,
    Bundles,
    Reduced_Model,
    Reduced_Encoder,
    Reduced_Decoder,
)
```
where
* `Unreduced_Model` is the block diagonal, but non-autonomous model.
* `Data_Encoder` is the transformation that brings the data into the block-diagonal form.
* `Data_Decoder` is the transformation that brings the model back into the coordinate system of the data.
* `Bundles` is a list of ranges that point out which entries of the `Unreduced_Model` are a vector bundle.
* `Reduced_Model` is the autonomous reduced model, if `Reduce == true`, otherwise same as `Unreduced_Model`.
* `Reduced_Encoder` brings `Unreduced_Model` into `Reduced_Model` if `Reduce == true` otherwise identity.
* `Reduced_Decoder` brings `Reduced_Model` back into `Unreduced_Model` if `Reduce == true` otherwise identity.

The parameter `By_Eigen` is true if the calculation is carried out by eigenvalue decomposition of the transport operator.
Otherwise a specially designed Hessenberg transformation and subsequent QR iteration is carried out on the vector bundles. This latter method
"""
function Create_Linear_Decomposition(
    BB_Filtered,
    SH;
    Time_Step = 1.0,
    Reduce = false,
    Align = true,
    By_Eigen = false,
)
    if By_Eigen
        Unreduced_Model,
        Data_Encoder,
        Data_Decoder,
        Reduced_Model,
        Reduced_Decoder,
        Reduced_Encoder,
        Bundles = Bundle_Decomposition_By_Eigenvectors(
            BB_Filtered,
            SH;
            Time_Step = Time_Step,
            sparse = false,
            dims = size(BB_Filtered, 1),
        )
    else
        Unreduced_Model, Data_Encoder, Data_Decoder, Bundles =
            Bundle_Decomposition(BB_Filtered, SH; Align = Align)
        if Reduce
            Reduced_Model, Reduced_Decoder, V_R_SH, Reduced_Encoder, W_R_SH =
                Reduce_Bundles(Unreduced_Model, Bundles, SH)
        end
    end
    if !Reduce
        Reduced_Model = deepcopy(Unreduced_Model)
        Reduced_Encoder = zero(Data_Encoder)
        Reduced_Decoder = zero(Data_Encoder)
        for p in axes(Reduced_Encoder, 1), q in axes(Reduced_Encoder, 2)
            Reduced_Encoder[p, q, p] = 1
            Reduced_Decoder[p, q, p] = 1
        end
    end
    #
    BB_Mean = dropdims(mean(Reduced_Model, dims = 2), dims = 2)
    println("Linear_Decompose: Eigenvalues")
    for bb in Bundles
        EV = eigvals(BB_Mean[bb, bb])
        if length(bb) == 2
            println(
                "[",
                bb[1],
                "-",
                bb[end],
                "]: Frequency ",
                abs(angle(EV[1])) / Time_Step,
                " [rad/s]; ",
                abs(angle(EV[1])) / (2 * pi * Time_Step),
                " [Hz]; Damping ",
                -log(abs.(EV[1])) ./ abs.(angle(EV[1])),
            )
        else
            println("[", bb[1], "]: Decay rate ", real(EV[1]))
        end
    end
    # TESTING
    #     Data_Decomposed, Data_Var = Decompose_Data(Index_List, Data, Encoded_Phase, SS, SH, Data_Encoder)
    #     for k in 2:length(Index_List)
    #         Data_Decomposed_R = view(Data_Decomposed, :, Index_List[k-1]+1:Index_List[k])
    #         Data_Var_R = view(Data_Var, :, Index_List[k-1]+1:Index_List[k])
    #         Encoded_Phase_R = view(Encoded_Phase, :, Index_List[k-1]+1:Index_List[k])
    #         @tullio res1[i, j] := BB_Linear[i, p, q] * Encoded_Phase_R[p, j] * Data_Var_R[q, j]
    #         @tullio res2[i, j] := Unreduced_Model[i, p, q] * Encoded_Phase_R[p, j] * Data_Decomposed_R[q, j]
    #         @show k, norm(Data_Var_R[:,2:end] - res1[:,1:end-1])
    #         @show k, norm(Data_Decomposed_R[:,2:end] - res2[:,1:end-1])
    #     end
    #
    return (
        Unreduced_Model = Unreduced_Model,
        Data_Encoder = Data_Encoder,
        Data_Decoder = Data_Decoder,
        Bundles = Bundles,
        Reduced_Model = Reduced_Model,
        Reduced_Encoder = Reduced_Encoder,
        Reduced_Decoder = Reduced_Decoder,
    )
end

@doc raw"""
    Select_Bundles_By_Energy(
        Index_List,
        Data,
        Encoded_Phase,
        Steady_State,
        SH,
        Decomp;
        How_Many,
        Ignore_Real = true,
    )

Takes the named tuple produced by [`Create_Linear_Decomposition`](@ref) and selects the most energetic `How_Many` vector bundles of the data.
Then produces a new named touple as in [`Create_Linear_Decomposition`](@ref), that only contains these most energetic vector bundles.
The input arguments are
* `Index_List`, `Data`, `Encoded_Phase` are the training data in the standard form.
* `Steady_State` is the steady state of the system estimated by [`Estimate_Linear_Model`](@ref).
* `SH` is the forcing dynamics as produced by [`Estimate_Linear_Model`](@ref).
* `Decomp` is the named tuple produced by [`Create_Linear_Decomposition`](@ref).
* `Ignore_Real` if `true` the method only returns two dimensional vector bundles.
    This is helpful when the data includes slowly varying `DC` shift with high energy that would be pick by the method otherwise.
"""
function Select_Bundles_By_Energy(
    Index_List,
    Data,
    Encoded_Phase,
    SS,
    SH,
    Decomp;
    How_Many,
    Ignore_Real = true,
    Time_Step = 1.0,
)
    if Ignore_Real
        Bundles = filter(x -> (length(x) == 2), Decomp.Bundles)
    else
        Bundles = Decomp.Bundles
    end
    Data_Encoder = Decomp.Data_Encoder
    Limit = min(length(Bundles), How_Many)
    Data_Var = Data - SS * Encoded_Phase
    @tullio Data_Decomposed[i, k] :=
        Data_Encoder[i, q, p] * Data_Var[p, k] * Encoded_Phase[q, k]
    Energies =
        [norm(Data_Decomposed[bb, :]) / norm(Data_Encoder[bb, :, :]) for bb in Bundles]
    Order = sortperm(Energies, rev = true)
    println("Bundles selected:")
    display(Bundles[Order[1:Limit]])
#     println("Energies:")
#     for k in Order[1:min(2 * Limit, length(Order))]
#         println(Bundles[k], " -> E=", Energies[k])
#     end
    Select = vcat(Bundles[Order[1:Limit]]...)
    if Select == range(first(Select), last(Select))
        Select = range(first(Select), last(Select))
    end
    Index = 1
    Re_Bundles = []
    for s in Order[1:Limit]
        push!(Re_Bundles, Bundles[s] .- (Bundles[s][1] - Index))
        Index = Re_Bundles[end][end] + 1
    end
#     println("Reconstituted bundles:")
#     display(Re_Bundles)
    BB_Mean = dropdims(mean(Decomp.Reduced_Model, dims = 2), dims = 2)
    println("Select_Bundles_By_Energy: Eigenvalues")
    for (bb, ee) in zip(Bundles[Order], Energies[Order])
        EV = eigvals(BB_Mean[bb, bb])
        if length(bb) == 2
            println(
                "[", bb[1], "-", bb[end],
                "]: E=", ee,
                " Frequency ", abs(angle(EV[1])) / Time_Step,
                " [rad/s]; ", abs(angle(EV[1])) / (2 * pi * Time_Step),
                " [Hz]; Damping ", -log(abs.(EV[1])) ./ abs.(angle(EV[1])),
                )
        else
            println("[", bb[1], "]: E=", ee, " Decay rate ", real(EV[1]))
        end
    end
    return (
        Unreduced_Model = Decomp.Unreduced_Model[Select, :, Select],
        Data_Encoder = Decomp.Data_Encoder[Select, :, :],
        Data_Decoder = Decomp.Data_Decoder[:, :, Select],
        Bundles = Re_Bundles,
        Reduced_Model = Decomp.Reduced_Model[Select, :, Select],
        Reduced_Encoder = Decomp.Reduced_Encoder[Select, :, Select],
        Reduced_Decoder = Decomp.Reduced_Decoder[Select, :, Select],
    )
end

@doc raw"""
    Decompose_Data(Index_List, Data, Encoded_Phase, Steady_State, SH, Data_Encoder)

Creates a decomposed and projected data set from the input `Index_List`, `Data`, `Encoded_Phase`.
The `Steady_State` and `SH` are produced by [`Estimate_Linear_Model`](@ref) and `Data_Encoder` id produced by either
[`Select_Bundles_By_Energy`](@ref) or [`Create_Linear_Decomposition`](@ref) directly.
"""
function Decompose_Data(Index_List, Data, Encoded_Phase, SS, SH, Data_Encoder)
    Data_Var = Data - SS * Encoded_Phase
    @tullio Data_Decomposed[i, k] :=
        Data_Encoder[i, q, p] * Data_Var[p, k] * Encoded_Phase[q, k]
    return Data_Decomposed, Data_Var
end

@doc raw"""
    Decomposed_Data_Scaling(Data_Decomp, Bundles)

Creates a scaling tensor which, when applied to `Data_Decomp` makes each vector bundle have the same maximum amplitude,
while also making the whole data set fit inside the unit ball.
"""
function Decomposed_Data_Scaling(Data_Decomp, Bundles)
    Data_Scale = zeros(eltype(Data_Decomp), size(Data_Decomp, 1), 1, 1)
    for Bundle in Bundles
        Bundle_Amplitude = zero(eltype(Data_Decomp))
        for k in axes(Data_Decomp, 2)
            Amplitude = zero(eltype(Data_Decomp))
            for i in Bundle
                Amplitude += Data_Decomp[i, k]^2
            end
            if Bundle_Amplitude < Amplitude
                Bundle_Amplitude = Amplitude
            end
        end
        Data_Scale[Bundle, :, :] .= 1 / sqrt(Bundle_Amplitude)
    end
    Overall_Amplitude = zero(eltype(Data_Decomp))
    for k in axes(Data_Decomp, 2)
        Amplitude = zero(eltype(Data_Decomp))
        for i in axes(Data_Decomp, 1)
            Amplitude += (Data_Scale[i] * Data_Decomp[i, k])^2
        end
        if Overall_Amplitude < Amplitude
            Overall_Amplitude = Amplitude
        end
    end
    Data_Scale_Overall = (1 - 2^(-8)) / sqrt(Overall_Amplitude)
    Data_Scale .*= Data_Scale_Overall
    return Data_Scale
end
