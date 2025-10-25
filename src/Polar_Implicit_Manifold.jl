struct Polar_Implicit_Manifold{
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    # WW[:,1,:,:] is constant, set as initial condition
    # TT[1] is constant, set as initial condition
    # RR[1] is zero (for the origin), set as initial condition
    # The scale in the radial direction is always[0, 1]
    Radius::Any
    Radial_Mesh::Any
    W0::Any   # 4 dimensional array : n_s x n_r x n_b x n_t
    Latent_Transformation::Any
    Radii::Any
end

function Take_Part(
    PP::Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    XX_Full,
    Intervals::Tuple{Integer,Integer},
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    XX_Part = deepcopy(XX_Full[:, start_index:final_index, :, :])
    return XX_Part
end

function Find_Implicit_Tangent(
    MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
    XTF,
    Torus,
    Index,
    Phase_Dimension;
    Latent_Transformation = Diagonal(I, 2),
    Radii = ones(2),
) where {M,State_Dimension,Skew_Dimension}
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    Latent_Indices = MTF.Indices[Index]
    Latent_Extended = zeros(eltype(XTF), State_Dimension, Skew_Dimension, Phase_Dimension)
    for q in axes(Latent_Extended, 3)
        ZZ =
            Latent_Transformation *
            [Radii[1] * cos(Beta_Grid[q]); Radii[2] * sin(Beta_Grid[q])]
        for p in axes(Latent_Extended, 2)
            Latent_Extended[Latent_Indices[1], p, q] = ZZ[1]
            Latent_Extended[Latent_Indices[2], p, q] = ZZ[2]
        end
    end
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Jac_Full = zeros(eltype(Torus), State_Dimension, State_Dimension, Skew_Dimension)
    for k = 1:M
        Jacobian!(
            view(Jac_Full, MTF.Indices[k], :, :),
            MTF[k][2],
            XTF.x[k].x[2],
            Torus,
            Encoded_Phase,
        )
    end
    Full_Tangent = zero(Latent_Extended)
    # the inversion
    for k in axes(Latent_Extended, 2)
        IJAC = pinv(Jac_Full[:, :, k])
        ev = abs.(eigvals(Jac_Full[:, :, k]))
        @show k, minimum(ev), maximum(ev)
        for l in axes(Latent_Extended, 3)
            Full_Tangent[:, k, l] .= IJAC * Latent_Extended[:, k, l]
        end
    end
    return Full_Tangent
end

function Polar_Implicit_Manifold(
    MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
    XTF,
    Index,
    Radial_Order,
    Radial_Intervals,
    Phase_Dimension,
    Radius;
    Latent_Data = Diagonal(I, 2),        #         Transformation = Diagonal(I, 2), Radii = ones(2)
) where {M,State_Dimension,Skew_Dimension}
    F = svd(Latent_Data * Latent_Data')
    Latent_Embed = F.Vt * Latent_Data
    Radii = vec(maximum(abs.(Latent_Embed), dims = 2))
    Latent_Transformation = F.U
    #
    Radial_Mesh = Chebyshev_Mesh(Radial_Order, Radial_Intervals)
    Radial_Dimension = length(Radial_Mesh)
    Torus = Find_Torus(MTF, XTF)
    println("Torus done")
    meminfo_julia()
    FT = Find_Implicit_Tangent(
        MTF,
        XTF,
        Torus,
        Index,
        Phase_Dimension;
        Latent_Transformation = Latent_Transformation,
        Radii = Radii,
    )
    println("Find_Implicit_Tangent done")
    meminfo_julia()
    WW = zeros(State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
    for k in axes(WW, 4), l in axes(WW, 2)
        WW[:, l, :, k] .= Torus
    end
    for k in eachindex(Radial_Mesh)
        WW[:, k, :, :] .+= FT .* Radial_Mesh[k] * Radius
    end
    println("WW done")
    meminfo_julia()
    Latent_Dimension = typeof(MTF[Index]).parameters[2]
    PP = Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    }(
        Radius,
        Radial_Mesh,
        deepcopy(WW[:, 1, :, :]),
        Latent_Transformation,
        Radii,
    )
    return PP, WW, Torus
end

function Evaluate!(
    Value,
    Jacobian,
    PP::Polar_Implicit_Manifold,
    XX_Full,
    Intervals::Tuple{Integer,Integer},
    XX_Part,
    MTF::Multi_Foliation,
    XTF,
    Select;
    Lambda = 1,
)
    return Evaluate!(
        Value,
        Jacobian,
        PP,
        XX_Full,
        Intervals,
        XX_Part,
        (x, y, z) -> Evaluate_Encoder_First!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
        (x, y, z) -> Evaluate_Encoder_Second!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
        (x, y, z) -> Jacobian_Encoder_First!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
        (x, y, z) -> Jacobian_Encoder_Second!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
    )
end

function Evaluate!(
    Value,
    Jacobian,
    PP::Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    XX_Full,
    Intervals::Tuple{Integer,Integer},
    XX_Part,
    Evaluate_First!::Function,
    Evaluate_Second!::Function,
    Jacobian_First!::Function,
    Jacobian_Second!::Function,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    Radius = PP.Radius
    Latent_Transformation = PP.Latent_Transformation
    Radii = PP.Radii
    #
    WW = zeros(
        eltype(XX_Part),
        State_Dimension,
        Radial_Dimension,
        Skew_Dimension,
        Phase_Dimension,
    )

    WW .= XX_Full
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    #     println("From = ", start_index, " To = ", final_index)
    #     @show size(WW[:, start_index:final_index, :, :]), size(XX_Part)
    WW[:, start_index:final_index, :, :] .= reshape(
        XX_Part,
        State_Dimension,
        1 + final_index - start_index,
        Skew_Dimension,
        Phase_Dimension,
    )

    Radial_Mesh = PP.Radial_Mesh
    MON_LIN, D_MON_LIN = Clenshaw_Interpolate(
        Radial_Mesh[start_index:final_index],
        Radial_Order,
        Radial_Mesh,
    )
    grid_r = Radial_Mesh[start_index:final_index] # disregard the first collocation point
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    #
    @tullio E_WW[s, r, p, q] := WW[s, j, p, q] * MON_LIN[r, j]
    E_WW_RS = reshape(E_WW, size(E_WW, 1), :)
    #
    E_Phase = zeros(Skew_Dimension, size(E_WW)[2:end]...)
    E_Latent = zeros(Latent_Dimension, size(E_WW)[2:end]...)
    for q in axes(E_Phase, 4)
        ZZ =
            Latent_Transformation *
            [Radii[1] * cos(Beta_Grid[q]); Radii[2] * sin(Beta_Grid[q])]
        for r in axes(E_Phase, 2), p in axes(E_Phase, 3)
            for s in axes(E_Phase, 1)
                E_Phase[s, r, p, q] = I[s, p]
            end
            E_Latent[1, r, p, q] = Radius * grid_r[r] * ZZ[1]
            E_Latent[2, r, p, q] = Radius * grid_r[r] * ZZ[2]
        end
    end
    E_Phase_RS = reshape(E_Phase, size(E_Phase, 1), :)
    #
    V1_R = range(1, Latent_Dimension * size(E_WW_RS, 2))
    V2_R = range(
        1 + last(V1_R),
        last(V1_R) + (State_Dimension - Latent_Dimension) * size(E_WW_RS, 2),
    )
    #
    if !isnothing(Value)
        Value .= 0
        #
        V1_E_WW_RS = zeros(eltype(WW), Latent_Dimension, size(E_WW_RS, 2))
        V2_E_WW_RS = zeros(eltype(WW), State_Dimension - Latent_Dimension, size(E_WW_RS, 2))
        Evaluate_First!(V1_E_WW_RS, E_WW_RS, E_Phase_RS)
        Evaluate_Second!(V2_E_WW_RS, E_WW_RS, E_Phase_RS)
        #
        V1_E_WW = reshape(V1_E_WW_RS, :, size(E_WW)[2:end]...)
        V2_E_WW = reshape(V2_E_WW_RS, :, size(E_WW)[2:end]...)

        Value[V1_R] .= vec((V1_E_WW .- E_Latent) ./ reshape(grid_r, 1, :, 1, 1))
        Value[V2_R] .= vec(V2_E_WW ./ reshape(grid_r, 1, :, 1, 1))
        if eltype(Value) <: Float64
            print(" E=", norm(Value), " E1=", maximum(abs.(Value[V1_R])))
            if State_Dimension != Latent_Dimension
                print(" E2=", maximum(abs.(Value[V2_R])))
            end
            println()
        end
    end

    ### JACOBIAN ###
    if !isnothing(Jacobian)
        Jacobian .= 0
        print(" J")
        MON_LIN_R = view(MON_LIN, :, start_index:final_index)
        #
        J_V1_E_WW_RS =
            zeros(eltype(WW), Latent_Dimension, State_Dimension, size(E_WW_RS, 2))
        J_V2_E_WW_RS = zeros(
            eltype(WW),
            State_Dimension - Latent_Dimension,
            State_Dimension,
            size(E_WW_RS, 2),
        )
        Jacobian_First!(J_V1_E_WW_RS, E_WW_RS, E_Phase_RS)
        Jacobian_Second!(J_V2_E_WW_RS, E_WW_RS, E_Phase_RS)
        J_FULL = cat(J_V1_E_WW_RS, J_V2_E_WW_RS, dims = 1)
        min_rat = typemax(eltype(J_FULL))
        for k in axes(J_FULL, 3)
            vals = abs.(eigvals(J_FULL[:, :, k]))
            if min_rat > minimum(vals) / maximum(vals)
                min_rat = minimum(vals) / maximum(vals)
            end
        end
        @show min_rat
        # Indices
        WW_C = range(1, length(XX_Part))
        # 1. V1 o W
        Jac_V1_WW = reshape(
            view(Jacobian, V1_R, WW_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            State_Dimension,
            size(E_WW)[2:end]...,
        )
        J_V1 = reshape(J_V1_E_WW_RS, :, State_Dimension, size(E_WW)[2:end]...)
        for l in axes(J_V1, 5),
            k in axes(J_V1, 4),
            j in axes(J_V1, 3),
            a in axes(J_V1, 2),
            p in axes(J_V1, 1),
            b in axes(MON_LIN_R, 2)

            Jac_V1_WW[p, j, k, l, a, b, k, l] += J_V1[p, a, j, k, l] * MON_LIN_R[j, b]
        end
        Jac_V1 = reshape(view(Jacobian, V1_R, :), Latent_Dimension, size(E_WW)[2:end]..., :)
        Jac_V1 ./= reshape(grid_r, 1, :, 1, 1, 1)
        # 2. V2 o W
        #         if State_Dimension != Latent_Dimension
        Jac_V2_WW = reshape(
            view(Jacobian, V2_R, WW_C),
            State_Dimension - Latent_Dimension,
            size(E_WW)[2:end]...,
            State_Dimension,
            size(E_WW)[2:end]...,
        )
        Jac_V2_WW .= 0
        J_V2 = reshape(J_V2_E_WW_RS, :, State_Dimension, size(E_WW)[2:end]...)
        for l in axes(J_V2, 5),
            k in axes(J_V2, 4),
            j in axes(J_V2, 3),
            a in axes(J_V2, 2),
            p in axes(J_V2, 1),
            b in axes(MON_LIN_R, 2)

            Jac_V2_WW[p, j, k, l, a, b, k, l] += J_V2[p, a, j, k, l] * MON_LIN_R[j, b]
        end
        Jac_V2 = reshape(
            view(Jacobian, V2_R, :),
            State_Dimension - Latent_Dimension,
            size(E_WW)[2:end]...,
            :,
        )
        Jac_V2 ./= reshape(grid_r, 1, :, 1, 1, 1)
        #         end
        if any(isnan.(Jacobian))
            print(" JNaN")
        end
    end
    return nothing
end

function Tangent_Extend!(
    PP::Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    XX,
    Last_Interval::Integer,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    Radial_Mesh = PP.Radial_Mesh
    WW = XX
    #
    final_index = 1 + Last_Interval * (Radial_Order - 1)
    IPT, DTP = Clenshaw_Interpolate(Radial_Mesh[1:final_index], Radial_Order, Radial_Mesh)
    D_MM = DTP[end, :]
    @tullio D_WW[i, k, l] := D_MM[s] * WW[i, s, k, l]
    D_WW_R = reshape(D_WW, size(D_WW, 1), 1, size(D_WW, 2), size(D_WW, 3))
    #
    Scaling = Radial_Mesh[(final_index+1):end] .- Radial_Mesh[final_index]
    WW[:, (final_index+1):end, :, :] .=
        WW[:, [final_index], :, :] .+ D_WW_R .* reshape(Scaling, 1, :, 1, 1)
    return nothing
end

function Set_Part!(
    PP::Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    XX_Full,
    Intervals::Tuple{Integer,Integer},
    XX_Part,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    XX_Full[:, start_index:final_index, :, :] .= reshape(
        XX_Part,
        State_Dimension,
        1 + final_index - start_index,
        Skew_Dimension,
        Phase_Dimension,
    )
    return nothing
end

# already in polarmanifold-cheb.jl
# function To_Latent(MTF::Multi_Foliation, XTF, Index, Data, Encoded_Phase)
#     Latent_Data = zeros(eltype(Data), size(XTF.x[Index].x[1].WW, 1), size(Data, 2))
#     Evaluate!(Latent_Data, MTF[Index][2], XTF.x[Index].x[2], Data, Encoded_Phase)
#     return Latent_Data
# end

function Embed_Manifold(
    PM::Polar_Implicit_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    WW,
    Latent_Data,
    Encoded_Phase;
    Output_Transformation = [
        I[i, j] for i = 1:State_Dimension, k = 1:Skew_Dimension, j = 1:State_Dimension
    ],
    Output_Inverse_Transformation = [],
    Output_Scale = ones(State_Dimension),
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    Radius = PM.Radius
    Latent_Transformation = PM.Latent_Transformation
    Radii = PM.Radii
    #
    Latent_Embed = transpose(Latent_Transformation) * Latent_Data
    Latent_Embed_Scaled = Diagonal(1 ./ Radii) * Latent_Embed
    Latent_Radius = sqrt.(vec(sum(Latent_Embed_Scaled .^ 2, dims = 1)))
    Latent_Angle = mod.(atan.(Latent_Embed_Scaled[2, :], Latent_Embed_Scaled[1, :]), 2 * pi)
    #     Latent_Angle = atan.(Latent_Embed_Scaled[2, :], Latent_Embed_Scaled[1, :])
    @show norm(Latent_Embed_Scaled[1, :] - Latent_Radius .* cos.(Latent_Angle))
    @show norm(Latent_Embed_Scaled[2, :] - Latent_Radius .* sin.(Latent_Angle))
    Valid_Ind = (Latent_Radius .<= Radius)
    #
    Radial_Mesh = PM.Radial_Mesh
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    #
    Ordering = sortperm(Latent_Radius)
    ITP_Ordered, _ =
        Clenshaw_Interpolate(Latent_Radius[Ordering] ./ Radius, Radial_Order, Radial_Mesh)
    ITP = ITP_Ordered[invperm(Ordering), :]
    #         Barycentric_Interpolation_Matrix(Radial_Order, Radial_Mesh, Latent_Radius ./ Radius)
    @tullio SH_beta[q, k] := psi(Latent_Angle[k] - Beta_Grid[q], Phase_Dimension)
    @show size(ITP), size(SH_beta), size(Encoded_Phase)
    @tullio E_WW_I[i, k] := ITP[k, r] * WW[i, r, j, q] * SH_beta[q, k] * Encoded_Phase[j, k]
    # The amplitude
    AA = zeros(eltype(E_WW_I), size(E_WW_I, 2))
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    if isempty(Output_Inverse_Transformation)
        E_WW = similar(E_WW_I)
        for k in axes(E_WW, 2)
            Phase = view(Encoded_Phase, :, k)
            @tullio TRMAT[i, j] := Output_Transformation[i, p, j] * Phase[p]
            INV_TRMAT = inv(TRMAT)
            E_WW[:, k] .= INV_TRMAT * (E_WW_I[:, k] ./ vec(Output_Scale))
            #
            ITP_R = view(ITP, k, :)
            @tullio WW_AMP[i, q, l] :=
                INV_TRMAT[i, p] * WW[p, j, q, l] * ITP_R[j] / Output_Scale[i]
            AA[k] = sqrt(dot(WW_AMP, WW_AMP) / Factor)
        end
    else
        E_WW =
            zeros(eltype(E_WW_I), size(Output_Inverse_Transformation, 1), size(E_WW_I, 2))
        for k in axes(E_WW, 2)
            Phase = view(Encoded_Phase, :, k)
            @tullio TRMAT[i, j] := Output_Inverse_Transformation[i, p, j] * Phase[p]
            E_WW[:, k] .= TRMAT * (E_WW_I[:, k] ./ vec(Output_Scale))
            #
            ITP_R = view(ITP, k, :)
            @tullio WW_AMP[i, q, l] :=
                TRMAT[i, p] * WW[p, j, q, l] * ITP_R[j] / Output_Scale[p]
            AA[k] = sqrt(dot(WW_AMP, WW_AMP) / Factor)
        end
    end
    return E_WW, E_WW_I, AA, Valid_Ind
end

"""
    Extract_Manifold_Embedding(
        MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
        XTF,
        Index,
        Data_Decomp,
        Encoded_Phase;
        Radial_Order,
        Radial_Intervals,
        Radius,
        Phase_Dimension,
        Output_Transformation = [],
        Output_Inverse_Transformation = [],
        Output_Scale,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 24,
        initial_maxiters = 200,
    )

Extracts the invariant manifold from a set of invariant foliations `MTF`, `XTF`.
The parametrisation of the foliation is the same as the identofied conjugate map.
Therefore it is not directly useable to obtain backbone curves.
The result is primarily used to find out the physical amplitude of an encoded data point that is mapped back onto the invariant manifold.

The input arguments are
* `MTF`, `XTF` represent the set of invariant foliations previously fitted to data
* `Index` chooses the invariant foliation for which we calculate the invariant manifold
* `Radial_Order` order of the Chebyshev in the radial direction
* `Radial_Intervals` number of polynomials in the radial direction
* `Radius` the maximum radius to calculate the invariant manifold for
* `Phase_Dimension` the number of Fourier collocation points to use in the angular direction
* `Output_Transformation` same as `Data_Encoder` produced by [`Create_Linear_Decomposition`](@ref).
* `Output_Inverse_Transformation` same as `Data_Decoder` produced by [`Create_Linear_Decomposition`](@ref).
    Note that only one of `Output_Transformation` or `Output_Inverse_Transformation` need to be specified.
    If `Output_Inverse_Transformation` is specified, it takes precedence.
* `Output_Scale` same as `Data_Scale` produced by [`Decomposed_Data_Scaling`](@ref)
* `abstol = 1e-9` absolute tolerance when solving the invariance equation
* `reltol = 1e-9` relative tolerance when solving the invariance equation
* `maxiters = 12` number of solution steps when calculating each segment of the manifold
* `initial_maxiters` the maximum iteration when calculating the segment containing the steady state.
    About the steady state the manifold is non-hyperbolic and therefore numerically challenging to calculate.
"""
function Extract_Manifold_Embedding(
    MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
    XTF,
    Index,
    Data_Decomp,
    Encoded_Phase;
    Radial_Order,
    Radial_Intervals,
    Radius,
    Phase_Dimension,
    Output_Transformation = [],
    Output_Inverse_Transformation = [],
    Output_Scale,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 24,
    initial_maxiters = 200,
) where {M,State_Dimension,Skew_Dimension}
    Latent_Data = To_Latent(MTF, XTF, Index, Data_Decomp, Encoded_Phase)
    #     F = svd(Latent_Data * Latent_Data')
    #     Latent_Embed = F.Vt * Latent_Data
    #     Radii = vec(maximum(abs.(Latent_Embed), dims=2))
    #     Latent_Transformation = F.U

    MIP, XIP, Torus = Polar_Implicit_Manifold(
        MTF,
        XTF,
        Index,
        Radial_Order,
        Radial_Intervals,
        Phase_Dimension,
        Radius,
        Latent_Data = Latent_Data,
    )

    for it = 1:Radial_Intervals
        println(
            "Implict: Interval = ",
            it,
            " of ",
            Radial_Intervals,
            " = ",
            it / Radial_Intervals,
        )
        Intervals = (it, it)
        XIP_Part = Take_Part(MIP, XIP, Intervals)
        fun = NonlinearFunction(
            (res, u, p) ->
                Evaluate!(res, nothing, MIP, XIP, Intervals, u, MTF, XTF, Index);
            jac = (J, u, p) ->
                Evaluate!(nothing, J, MIP, XIP, Intervals, u, MTF, XTF, Index),
        )
        prob = NonlinearProblem(fun, XIP_Part)
        sol = solve(
            prob,
            NonlinearSolve.NLsolveJL(),
            abstol = abstol,
            reltol = reltol,
            maxiters = ifelse(it == 1, initial_maxiters, maxiters),
        )
        if (it > 1) && !SciMLBase.successful_retcode(sol)
            break
        end
        Set_Part!(MIP, XIP, Intervals, sol.u)
        Tangent_Extend!(MIP, XIP, it)
    end

    E_WW, E_WW_I, AA, Valid_Ind = Embed_Manifold(
        MIP,
        XIP,
        Latent_Data,
        Encoded_Phase;
        Output_Transformation = Output_Transformation,
        Output_Inverse_Transformation = Output_Inverse_Transformation,
        Output_Scale = Output_Scale,
    )
    E_ENC = similar(E_WW_I)
    Evaluate_Encoder_First!(
        view(E_ENC, 1:size(Latent_Data, 1), :),
        MTF,
        XTF,
        Index,
        E_WW_I,
        Encoded_Phase;
        Lambda = 1,
    )
    Evaluate_Encoder_Second!(
        view(E_ENC, (1+size(Latent_Data, 1)):size(E_ENC, 1), :),
        MTF,
        XTF,
        Index,
        E_WW_I,
        Encoded_Phase;
        Lambda = 1,
    )
    println("The moment of truth")
    @show norm(Latent_Data - view(E_ENC, 1:size(Latent_Data, 1), :))
    @show norm(view(E_ENC, (1+size(Latent_Data, 1)):size(E_ENC, 1), :))
    return MIP, XIP, Torus, E_WW, Latent_Data, E_ENC, AA, Valid_Ind
end
