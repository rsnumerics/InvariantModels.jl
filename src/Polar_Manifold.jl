# 1. Evaluate the equations on the grid
# 1.a Create the grid
# 1.b Evaluate T() and R() on the grid
# 1.c Evaluate W() on the (transformed) grid
# 1.d Evaluate D_1 W() and D_2 W() on the grid
# 2. W has a W_0 component for the constant part and W_n component

# -> W on the grid
# -> W on the shifted grid (R(r), beta+T(r),t+omega)
# -> W_par on W on grid
# -> W_
# -> F() on the grid QPModel

# In the Skew_Dimenesion, we need the shift operator, as opposed to an Omega

struct Polar_Manifold{
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
    Radial_Mesh::Any
    M_Model::Any
    X_Model::Any
    SH::Any   # shift operator in theta (n_t)
    W0::Any   # 4 dimensional array : n_s x n_r x n_b x n_t
    R0::Any   # 1 dimensional array : n_r  (polynomial coefficients)
    T0::Any   # 1 dimensional array : n_r  (polynomial coefficients)
    Transformation::Any
end

# function Polar_Manifold(State_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension, Transformation)
#     # WW = zeros(State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
#     # RR = zeros(Radial_Dimension)
#     # TT = zeros(Radial_Dimension)
#     W0 = zeros(State_Dimension, Skew_Dimension, Phase_Dimension)
#     R0 = zero(1)
#     T0 = zero(1)
#     return Polar_Manifold{State_Dimension, Latent_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension}(WW, RR, TT, Transformation)
# end

function Evaluate_Encoder_First!(
    Result,
    MTF::Multi_Foliation,
    XTF,
    Select,
    Data,
    Encoded_Phase;
    Lambda = 1,
)
    return Evaluate!(
        Result,
        MTF[Select][2],
        XTF.x[Select].x[2],
        Data,
        Encoded_Phase;
        Lambda = Lambda,
    )
end

function Evaluate_Encoder_Second!(
    Result,
    MTF::Multi_Foliation,
    XTF,
    Select,
    Data,
    Encoded_Phase;
    Lambda = 1,
)
    it::Int = 0
    for k = 1:length(MTF)
        if k != Select
            Local_Latent = typeof(MTF[k]).parameters[2]
            Evaluate!(
                view(Result, (it+1):(it+Local_Latent), :),
                MTF[k][2],
                XTF.x[k].x[2],
                Data,
                Encoded_Phase;
                Lambda = Lambda,
            )
            it += Local_Latent
        end
    end
    return Result
end

function Jacobian_Encoder_First!(
    Result,
    MTF::Multi_Foliation,
    XTF,
    Select,
    Data,
    Encoded_Phase;
    Lambda = 1,
)
    return Jacobian!(
        Result,
        MTF[Select][2],
        XTF.x[Select].x[2],
        Data,
        Encoded_Phase;
        Lambda = Lambda,
    )
end

function Jacobian_Encoder_Second!(
    Result,
    MTF::Multi_Foliation,
    XTF,
    Select,
    Data,
    Encoded_Phase;
    Lambda = 1,
)
    it::Int = 0
    for k = 1:length(MTF)
        if k != Select
            Local_Latent = typeof(MTF[k]).parameters[2]
            Jacobian!(
                view(Result, (it+1):(it+Local_Latent), :, :),
                MTF[k][2],
                XTF.x[k].x[2],
                Data,
                Encoded_Phase;
                Lambda = Lambda,
            )
            it += Local_Latent
        end
    end
    return Result
end

function Evaluate!(
    Value,
    Jacobian,
    PP::Polar_Manifold,
    XX_Full,
    Intervals::Tuple{Integer,Integer},
    XX_Part,
    Radius,
)
    return Evaluate!(
        Value,
        Jacobian,
        PP,
        XX_Full,
        Intervals,
        XX_Part,
        (x, y, z) -> copyto!(x, y),
        (x, y, z) -> nothing,
        (x, y, z) -> begin
            for i in axes(x, 1), j in axes(x, 2), k in axes(x, 3)
                x[i, j, k] = I[i, j]
            end
            nothing
        end,
        (x, y, z) -> nothing,
        Radius,
    )
end

function Evaluate!(
    Value,
    Jacobian,
    PP::Polar_Manifold,
    XX_Full,
    Intervals::Tuple{Integer,Integer},
    XX_Part,
    MTF::Multi_Foliation,
    XTF,
    Select,
    Radius;
    Lambda = 1,
)
    return Evaluate!(
        Value,
        Jacobian,
        PP,
        XX_Full,
        Intervals,
        XX_Part,
        (x, y, z) -> Evaluate_Encoder_First!(x, MTF, XTF, Select, y, z),
        (x, y, z) -> Evaluate_Encoder_Second!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
        (x, y, z) -> Jacobian_Encoder_First!(x, MTF, XTF, Select, y, z),
        (x, y, z) -> Jacobian_Encoder_Second!(x, MTF, XTF, Select, y, z; Lambda = Lambda),
        Radius,
    )
end

function Take_Part(
    PP::Polar_Manifold{
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
    CA = ComponentArray(
        WW = deepcopy(XX_Full.WW[:, start_index:final_index, :, :]),
        RR = deepcopy(XX_Full.RR[start_index:final_index]),
        TT = deepcopy(XX_Full.TT[start_index:final_index]),
    )
    return CA
end

function Set_Part!(
    PP::Polar_Manifold{
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
    WW = XX_Full.WW
    RR = XX_Full.RR
    TT = XX_Full.TT
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    WW[:, start_index:final_index, :, :] .= XX_Part.WW
    RR[start_index:final_index] .= XX_Part.RR
    TT[start_index:final_index] .= XX_Part.TT
    return nothing
end

function Tangent_Extend!(
    PP::Polar_Manifold{
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
    WW = XX.WW
    RR = XX.RR
    TT = XX.TT
    #
    final_index = 1 + Last_Interval * (Radial_Order - 1)
    IPT, DTP = Clenshaw_Interpolate(Radial_Mesh[1:final_index], Radial_Order, Radial_Mesh)
    D_RR = dot(DTP[end, :], RR)
    D_TT = dot(DTP[end, :], TT)
    D_MM = DTP[end, :]
    @tullio D_WW[i, k, l] := D_MM[s] * WW[i, s, k, l]
    D_WW_R = reshape(D_WW, size(D_WW, 1), 1, size(D_WW, 2), size(D_WW, 3))
    #
    Scaling = Radial_Mesh[(final_index+1):end] .- Radial_Mesh[final_index]
    RR[(final_index+1):end] .= RR[final_index] .+ D_RR * Scaling
    TT[(final_index+1):end] .= TT[final_index] .+ D_TT * Scaling
    WW[:, (final_index+1):end, :, :] .=
        WW[:, [final_index], :, :] .+ D_WW_R .* reshape(Scaling, 1, :, 1, 1)
    return nothing
end

function Evaluate!(
    Value,
    Jacobian,
    PP::Polar_Manifold{
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
    Radius;
    Arclength::Bool = false,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    WW = zeros(
        eltype(XX_Part),
        State_Dimension,
        Radial_Dimension,
        Skew_Dimension,
        Phase_Dimension,
    )
    RR = zeros(eltype(XX_Part), Radial_Dimension)
    TT = zeros(eltype(XX_Part), Radial_Dimension)

    WW .= XX_Full.WW
    RR .= XX_Full.RR
    TT .= XX_Full.TT
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    #     println("From = ", start_index, " To = ", final_index)
    WW[:, start_index:final_index, :, :] .= XX_Part.WW
    RR[start_index:final_index] .= XX_Part.RR
    TT[start_index:final_index] .= XX_Part.TT
    #
    SH = PP.SH
    M_Model, X_Model = PP.M_Model, PP.X_Model

    Radial_Mesh = PP.Radial_Mesh
    MON_LIN, D_MON_LIN = Clenshaw_Interpolate(
        Radial_Mesh[start_index:final_index],
        Radial_Order,
        Radial_Mesh,
    )
    grid_r = Radial_Mesh[start_index:final_index] # disregard the first collocation point
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    #
    radii = RR[start_index:final_index]
    shift = TT[start_index:final_index]
    MON_RR, DMR =
        Clenshaw_Interpolate(RR[start_index:final_index], Radial_Order, Radial_Mesh)
    #     @show size(DMR), size(MON_LIN)
    @tullio D_MON_RR[r, s, p] := DMR[r, s] * MON_LIN[r, p]
    #
    @tullio E_WW[s, r, p, q] := WW[s, j, p, q] * MON_LIN[r, j]
    E_WW_RS = reshape(E_WW, size(E_WW, 1), :)
    #
    @tullio SH_beta[q, r, k] := psi(Beta_Grid[q] + shift[r] - Beta_Grid[k], Phase_Dimension)
    @tullio D_SH_beta[q, r, k, p] :=
        D_psi(Beta_Grid[q] + shift[r] - Beta_Grid[k], Phase_Dimension) * MON_LIN[r, p]

    @tullio E_WW_RT[i, r, j, k] := WW[i, s, j, q] * MON_RR[r, s] * SH_beta[k, r, q]
    E_WW_RT_RS = reshape(E_WW_RT, size(E_WW_RT, 1), :)
    #
    E_Phase = zeros(Skew_Dimension, size(E_WW)[2:end]...)
    for s in axes(E_Phase, 1),
        r in axes(E_Phase, 2),
        p in axes(E_Phase, 3),
        q in axes(E_Phase, 4)

        E_Phase[s, r, p, q] = I[s, p]
    end
    E_Phase_RS = reshape(E_Phase, size(E_Phase, 1), :)
    V1_E_WW_RS = zeros(eltype(WW), Latent_Dimension, size(E_WW_RS, 2))
    Evaluate_First!(V1_E_WW_RS, E_WW_RS, E_Phase_RS)
    #
    WWN = deepcopy(WW)
    for k in axes(WWN, 2)
        WWN[:, k, :, :] .-= WW[:, 1, :, :]
    end
    # this is a quadratic form with the transformation
    Transformation = PP.Transformation
    @tullio WWP[j, r, k, l] :=
        Transformation[i, k, j] * Transformation[i, k, p] * WWN[p, r, k, l]
    #     WWP[1+Latent_Dimension:end, :, : , :] .= 0
    DD = Fourier_Differential(Phase_Dimension)
    #
    V1_R = range(1, Latent_Dimension * size(E_WW_RS, 2))
    V2_R = range(
        1 + last(V1_R),
        last(V1_R) + (State_Dimension - Latent_Dimension) * size(E_WW_RS, 2),
    )
    RR_R = range(1 + last(V2_R), last(V2_R) + length(XX_Part.RR))
    TT_R = range(1 + last(RR_R), last(RR_R) + length(XX_Part.TT))
    #
    if !isnothing(Value)
        Value .= 0
        #
        V1_E_WW_RT_RS = zeros(eltype(WW), Latent_Dimension, size(E_WW_RT_RS, 2))
        V2_E_WW_RS = zeros(eltype(WW), State_Dimension - Latent_Dimension, size(E_WW_RS, 2))
        F_WW_RS = zero(V1_E_WW_RS)
        Evaluate_First!(V1_E_WW_RT_RS, E_WW_RT_RS, E_Phase_RS)
        Evaluate_Second!(V2_E_WW_RS, E_WW_RS, E_Phase_RS)
        Evaluate_Function!(F_WW_RS, M_Model, X_Model, V1_E_WW_RS, E_Phase_RS)
        #
        V1_E_WW = reshape(V1_E_WW_RS, :, size(E_WW)[2:end]...)
        V1_E_WW_RT = reshape(V1_E_WW_RT_RS, :, size(E_WW_RT)[2:end]...)
        V2_E_WW = reshape(V2_E_WW_RS, :, size(E_WW)[2:end]...)
        F_WW = reshape(F_WW_RS, :, size(E_WW)[2:end]...)
        # shift
        @show size(SH), size(V1_E_WW_RT)
        @tullio V1_E_WW_RT_SH[i, j, k, l] := V1_E_WW_RT[i, j, p, l] * SH[p, k]
        # PC
        @tullio D0WW[i, r, k, l] := WWP[i, j, k, l] * MON_LIN[r, j]
        @tullio D2WW[i, r, k, l] := WWP[i, j, k, p] * DD[p, l] * MON_LIN[r, j]
        @tullio D1WW[i, r, k, l] := WW[i, j, k, l] * D_MON_LIN[r, j]
        @tullio D1WWP[i, r, k, l] := WWP[i, j, k, l] * D_MON_LIN[r, j]
        Factor = (Phase_Dimension * Skew_Dimension / 2)
        #     @show grid_r .* Radius ^ 2 * Factor, vec(sum(D0WW .* D1WW, dims=(1,3,4))), sum(WWP[:, 2, :, :] .^ 2)
        #     Constraint_RR = vec(sum(D0WW .* D1WW, dims=(1,3,4))) .- (grid_r .* Radius ^ 2 * Factor)
        #     Constraint_TT = vec(sum(D1WW .* D2WW, dims=(1,3,4)))

        Value[V1_R] .= vec((V1_E_WW_RT_SH .- F_WW) ./ reshape(grid_r, 1, :, 1, 1))
        Value[V2_R] .= vec(V2_E_WW ./ reshape(grid_r, 1, :, 1, 1))
        if !Arclength
            Value[RR_R] .=
                vec(sum(D0WW .* D1WW, dims = (1, 3, 4))) ./ grid_r .- (Radius^2 * Factor)
        else
            Value[RR_R] .= vec(sum(D1WW .* D1WWP, dims = (1, 3, 4))) .- (Radius^2 * Factor)
        end
        Value[TT_R] .= vec(sum(D1WW .* D2WW, dims = (1, 3, 4))) ./ grid_r
        if eltype(Value) <: Float64
            print(" E=", norm(Value), " E1=", maximum(abs.(Value[V1_R])))
            if State_Dimension != Latent_Dimension
                print(" E2=", maximum(abs.(Value[V2_R])))
            end
            println(
                " E_R=",
                maximum(abs.(Value[RR_R])),
                " E_T=",
                maximum(abs.(Value[TT_R])),
            )
            #             println(" R=")
            #             display(vec(sum(D0WW .* D1WW, dims=(1,3,4))) ./ grid_r)
            #             println("T=")
            #             display(Value[TT_R])
        end
    end

    ### JACOBIAN ###
    if !isnothing(Jacobian)
        Jacobian .= 0
        print(" J")
        #         WWS = XX.WW
        MON_LIN_R = view(MON_LIN, :, start_index:final_index)
        D_MON_LIN_R = view(D_MON_LIN, :, start_index:final_index)
        #
        J_V1_E_WW_RS =
            zeros(eltype(WW), Latent_Dimension, State_Dimension, size(E_WW_RS, 2))
        J_V1_E_WW_RT_RS =
            zeros(eltype(WW), Latent_Dimension, State_Dimension, size(E_WW_RT_RS, 2))
        J_V2_E_WW_RS = zeros(
            eltype(WW),
            State_Dimension - Latent_Dimension,
            State_Dimension,
            size(E_WW_RS, 2),
        )
        J_F_WW_RS = zero(J_V1_E_WW_RS)
        Jacobian_First!(J_V1_E_WW_RS, E_WW_RS, E_Phase_RS)
        Jacobian_First!(J_V1_E_WW_RT_RS, E_WW_RT_RS, E_Phase_RS)
        Jacobian_Second!(J_V2_E_WW_RS, E_WW_RS, E_Phase_RS)
        Jacobian_Function!(J_F_WW_RS, M_Model, X_Model, V1_E_WW_RS, E_Phase_RS)
        #
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
        WW_C = range(1, length(XX_Part.WW))
        RR_C = range(1 + last(WW_C), last(WW_C) + length(XX_Part.RR))
        TT_C = range(1 + last(RR_C), last(RR_C) + length(XX_Part.TT))
        # 1.a
        Jac_V1_WW = reshape(
            view(Jacobian, V1_R, WW_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            State_Dimension,
            size(E_WW)[2:end]...,
        )
        # 1.a.V1 DW : SH o V1 o W o (R, T)
        MON_RR_R = view(MON_RR, :, start_index:final_index)
        DSK = Diagonal(I, Skew_Dimension)
        J_V1_RT = reshape(J_V1_E_WW_RT_RS, :, State_Dimension, size(E_WW)[2:end]...)
        @tullio Jac_V1_WW[p, j, k, l, a, b, c, d] +=
            (J_V1_RT[p, a, j, q, l] * DSK[q, c] * MON_RR_R[j, b] * SH_beta[l, j, d]) *
            SH[q, k]
        # 1.a.F
        J_F = reshape(J_F_WW_RS, :, State_Dimension, size(E_WW)[2:end]...)
        J_V1 = reshape(J_V1_E_WW_RS, :, State_Dimension, size(E_WW)[2:end]...)
        for l in axes(J_F, 5),
            k in axes(J_F, 4),
            j in axes(J_F, 3),
            a in axes(J_F, 2),
            p in axes(J_F, 1),
            b in axes(MON_LIN_R, 2),
            q in axes(J_V1, 1)

            Jac_V1_WW[p, j, k, l, a, b, k, l] -=
                J_F[p, q, j, k, l] * J_V1[q, a, j, k, l] * MON_LIN_R[j, b]
        end
        # 1.b. D_RR : SH o V1 o W o (R, T)
        D_MON_RR_R = view(D_MON_RR, :, :, start_index:final_index)
        Jac_V1_RR = reshape(
            view(Jacobian, V1_R, RR_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            length(XX_Part.RR),
        )
        @tullio Jac_V1_RR[i, j, k, l, p] =
            (
                J_V1_RT[i, a, j, q, l] *
                WW[a, b, c, d] *
                DSK[q, c] *
                D_MON_RR_R[j, b, p] *
                SH_beta[l, j, d]
            ) * SH[q, k]
        # 1.c. D_TT : SH o V1 o W o (R, T)
        D_SH_beta_R = view(D_SH_beta, :, :, :, start_index:final_index)
        Jac_V1_TT = reshape(
            view(Jacobian, V1_R, TT_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            length(XX_Part.RR),
        )
        @tullio Jac_V1_TT[i, j, k, l, p] =
            (
                J_V1_RT[i, a, j, q, l] *
                WW[a, b, c, d] *
                DSK[q, c] *
                MON_RR[j, b] *
                D_SH_beta_R[l, j, d, p]
            ) * SH[q, k]
        # rescale
        Jac_V1 = reshape(view(Jacobian, V1_R, :), Latent_Dimension, size(E_WW)[2:end]..., :)
        Jac_V1 ./= reshape(grid_r, 1, :, 1, 1, 1)
        # 2. V2 o W
        if State_Dimension != Latent_Dimension
            Jac_V2_WW = reshape(
                view(Jacobian, V2_R, WW_C),
                State_Dimension - Latent_Dimension,
                size(E_WW)[2:end]...,
                State_Dimension,
                size(E_WW)[2:end]...,
            )
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
        end
        # 3. D0WW .* D1WW
        Jac_RR_WW =
            reshape(view(Jacobian, RR_R, WW_C), length(XX_Part.RR), size(XX_Part.WW)...)
        Jac_TT_WW =
            reshape(view(Jacobian, TT_R, WW_C), length(XX_Part.TT), size(XX_Part.WW)...)

        #     @tullio Constraint_RR_3[r] := WWS[i, j1, k, l] * MON_LIN_R[r, j1] * WWS[i, j2, k, l] * D_MON_LIN_R[r, j2]
        #         @show size(Jac_RR_WW), size(MON_LIN_R), size(WWP), size(D_MON_LIN)
        if !Arclength
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] :=
                MON_LIN_R[r, b] * WWP[a, j2, d, g] * D_MON_LIN[r, j2]
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] +=
                WWP[a, j1, d, g] * MON_LIN[r, j1] * D_MON_LIN_R[r, b]
            Jac_RR_WW_tmp ./= reshape(grid_r, :, 1, 1, 1, 1)
        else
            # WITH DW * DW
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] :=
                2 * D_MON_LIN_R[r, b] * WWP[a, j2, d, g] * D_MON_LIN[r, j2]
        end
        Jac_RR_WW .= Jac_RR_WW_tmp
        #
        #     @tullio Constraint_TT_3[r] := WW[i, j1, k, p] * DD[p, l] * MON_LIN[r, j1] * WW[i, j2, k, l] * D_MON_LIN[r, j2]
        @tullio Jac_TT_WW_tmp[r, a, b, d, g] :=
            DD[g, l] * MON_LIN_R[r, b] * WWP[a, j2, d, l] * D_MON_LIN[r, j2]
        @tullio Jac_TT_WW_tmp[r, a, b, d, g] +=
            WWP[a, j1, d, p] * DD[p, g] * MON_LIN[r, j1] * D_MON_LIN_R[r, b]
        Jac_TT_WW .= Jac_TT_WW_tmp
        Jac_TT_WW ./= reshape(grid_r, :, 1, 1, 1, 1)
        if any(isnan.(Jacobian))
            print(" JNaN")
        end
    end
    return nothing
end

function Polar_Manifold(
    MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
    XTF,
    SH,
    Index,
    Radial_Order,
    Radial_Intervals,
    Phase_Dimension,
    Radius;
    Transformation = [
        I[i, j] for i = 1:State_Dimension, k = 1:Skew_Dimension, j = 1:State_Dimension
    ],
    Select = [1; 2],
) where {M,State_Dimension,Skew_Dimension}
    Radial_Mesh = Chebyshev_Mesh(Radial_Order, Radial_Intervals)
    Radial_Dimension = length(Radial_Mesh)
    if size(XTF.x[Index].x[1].WW, 2) == 1
        println("Making it Non Autonomous")
        M_Model, X_Model =
            To_Non_Autonomous(MTF[Index][1], XTF.x[Index].x[1], SH, Skew_Dimension)
    else
        M_Model, X_Model = MTF[Index][1], XTF.x[Index].x[1]
    end
    @show X_Model
    # same SH in all models
    #     SH = M_Model.SH
    Torus = Find_Torus(MTF, XTF)
    WW = zeros(State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
    RR = zeros(Radial_Dimension)
    TT = zeros(Radial_Dimension)
    # FT = zeros(State_Dimension, Skew_Dimension, Phase_Dimension)
    FT_tmp, DR, T0 = Find_Tangent(
        MTF,
        XTF,
        M_Model,
        X_Model,
        Index,
        Torus,
        Fourier_Grid(Phase_Dimension);
        Select = Select,
    )
    @tullio FT[i, k, l] := Transformation[i, k, p] * FT_tmp[p, k, l]
    @show Torus
    @show FT_tmp
    @show FT
    for k in axes(WW, 4), l in axes(WW, 2)
        #         @show size(WW[:, 1, :, k]), size(Torus)
        WW[:, l, :, k] .= Torus
    end
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    #     @show size(WW[:, 2, :, :]), size(FT)
    Amplitude = sqrt(sum(FT .* FT) / Factor)
    @show Factor, Amplitude, sqrt(sum(FT_tmp .* FT_tmp) / Factor)
    for k in eachindex(Radial_Mesh)
        WW[:, k, :, :] .+= FT_tmp .* (Radial_Mesh[k] * Radius / Amplitude)
    end
    #     @show sqrt(sum(WW[:, 2, :, :] .^ 2) / Factor)
    #     @show norm(WW[:, 2, :, :] )
    RR .= DR .* Radial_Mesh
    TT .= T0
    Latent_Dimension = size(X_Model.WW, 1)
    PP = Polar_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    }(
        Radial_Mesh,
        M_Model,
        X_Model,
        SH,
        deepcopy(WW[:, 1, :, :]),
        RR[1],
        TT[1],
        Transformation,
    )
    XX = ComponentArray(WW = WW, RR = RR, TT = TT)
    return PP, XX
end

function Polar_Manifold(
    M_Model::MultiStep_Model{
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        Trajectories,
    },
    X_Model,
    Select,
    Radial_Order,
    Radial_Intervals,
    Phase_Dimension,
    Radius;
    Transformation = [
        I[i, j] for i = 1:State_Dimension, k = 1:Skew_Dimension, j = 1:State_Dimension
    ],
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    Radial_Mesh = Chebyshev_Mesh(Radial_Order, Radial_Intervals)
    Radial_Dimension = length(Radial_Mesh)
    Torus, Jac = Find_Torus(M_Model, X_Model)
    WW = zeros(State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
    RR = zeros(Radial_Dimension)
    TT = zeros(Radial_Dimension)
    # FT = zeros(State_Dimension, Skew_Dimension, Phase_Dimension)
    FT_tmp, DR, T0 = Find_Tangent(M_Model, Jac, Select, Fourier_Grid(Phase_Dimension))
    @tullio FT[i, k, l] := Transformation[i, k, p] * FT_tmp[p, k, l]
    #     @show FT
    for k in axes(WW, 4), l in axes(WW, 2)
        #         @show size(WW[:, 1, :, k]), size(Torus)
        WW[:, l, :, k] .= Torus
    end
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    #     @show size(WW[:, 2, :, :]), size(FT)
    Amplitude = sqrt(sum(FT .* FT) / Factor)
    @show Amplitude, sqrt(sum(FT_tmp .* FT_tmp) / Factor)
    for k in eachindex(Radial_Mesh)
        WW[:, k, :, :] .+= FT_tmp .* (Radial_Mesh[k] * Radius / Amplitude)
    end
    #     @show sqrt(sum(WW[:, 2, :, :] .^ 2) / Factor)
    #     @show norm(WW[:, 2, :, :] )
    RR .= DR .* Radial_Mesh
    TT .= T0
    PP = Polar_Manifold{
        State_Dimension,
        State_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    }(
        Radial_Mesh,
        M_Model,
        X_Model,
        M_Model.SH,
        deepcopy(WW[:, 1, :, :]),
        RR[1],
        TT[1],
        Transformation,
    )
    XX = ComponentArray(WW = WW, RR = RR, TT = TT)
    return PP, XX
end

"""
    Find_MAP_Manifold(
        MM::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
        MX,
        Select;
        Radial_Order,
        Radial_Intervals,
        Radius,
        Phase_Dimension,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 12,
        initial_maxiters = 200,
    ) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}

Calculates a two-dimensional invariant manifold from the map `MM`, `MX`.
The result is stored as a piecewise Chebyshev polynomial in the radial direction Fourier collocation in the angular direction.

The input arguments are
* `MM` and `MX` are the discrete-time map
* `Select` chooses along which vector bundle to calculate the invariant manifold
* `Radial_Order` order of the Chebyshev in the radial direction
* `Radial_Intervals` number of polynomials in the radial direction
* `Radius` the maximum radius to calculate the invariant manifold for
* `Phase_Dimension` the number of Fourier collocation points to use in the angular direction
* `abstol = 1e-9` absolute tolerance when solving the invariance equation
* `reltol = 1e-9` relative tolerance when solving the invariance equation
* `maxiters = 12` number of solution steps when calculating each segment of the manifold
* `initial_maxiters` the maximum iteration when calculating the segment containing the steady state.
    About the steady state the manifold is non-hyperbolic and therefore numerically challenging to calculate.
"""
function Find_MAP_Manifold(
    MM::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    MX,
    Select;
    Radial_Order,
    Radial_Intervals,
    Radius,
    Phase_Dimension,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 12,
    initial_maxiters = 200,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    PM, PX = Polar_Manifold(
        MM,
        MX,
        Select,
        Radial_Order,
        Radial_Intervals,
        Phase_Dimension,
        Radius,
    )
    for it = 1:Radial_Intervals
        println(
            "MAP: Interval = ",
            it,
            " of ",
            Radial_Intervals,
            " = ",
            it / Radial_Intervals,
        )
        Intervals = (it, it)
        PX_Part = Take_Part(PM, PX, Intervals)
        fun = NonlinearFunction(
            (res, u, p) -> Evaluate!(res, nothing, PM, PX, Intervals, u, Radius);
            jac = (J, u, p) -> Evaluate!(nothing, J, PM, PX, Intervals, u, Radius),
        )
        prob = NonlinearProblem(fun, PX_Part)
        sol = solve(
            prob,
            NonlinearSolve.NLsolveJL(),
            abstol = abstol,
            reltol = reltol,
            maxiters = ifelse(it == 1, initial_maxiters, maxiters),
        )
        if (it > 1) && !SciMLBase.successful_retcode(sol)
            start_index = 2 + (it - 1) * (Radial_Order - 1)
            PX.WW[:, start_index:end, :, :] .= NaN
            PX.RR[start_index:end] .= NaN
            PX.TT[start_index:end] .= NaN
            break
        end
        Set_Part!(PM, PX, Intervals, sol.u)
        Tangent_Extend!(PM, PX, it)
    end
    return PM, PX
end

"""
    Find_DATA_Manifold(MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension}, XTF, SH, Index;
        Radial_Order, Radial_Intervals, Radius,
        Phase_Dimension, Transformation,
        abstol=1e-9, reltol=1e-9, maxiters=12, initial_maxiters=200
    ) where {M,State_Dimension,Skew_Dimension}

Calculates a two-dimensional invariant manifold from the set of invariant foliations `MTF`, `XTF`.
The calulation is for the invariant foliation selected by `Index`. `SH` is the forcing map.
The rest of the function arguments are the same as in [`Find_MAP_Manifold`](@ref)
"""
function Find_DATA_Manifold(
    MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
    XTF,
    SH,
    Index;
    Radial_Order,
    Radial_Intervals,
    Radius,
    Phase_Dimension,
    Transformation,
    abstol = 1e-9,
    reltol = 1e-9,
    maxiters = 12,
    initial_maxiters = 200,
) where {M,State_Dimension,Skew_Dimension}
    PM, PX = Polar_Manifold(
        MTF,
        XTF,
        SH,
        Index,
        Radial_Order,
        Radial_Intervals,
        Phase_Dimension,
        Radius,
        Transformation = Transformation,
    )
    for it = 1:Radial_Intervals
        println(
            "DATA: Interval = ",
            it,
            " of ",
            Radial_Intervals,
            " = ",
            it / Radial_Intervals,
        )
        Intervals = (it, it)
        PX_Part = Take_Part(PM, PX, Intervals)
        fun = NonlinearFunction(
            (res, u, p) ->
                Evaluate!(res, nothing, PM, PX, Intervals, u, MTF, XTF, Index, Radius);
            jac = (J, u, p) ->
                Evaluate!(nothing, J, PM, PX, Intervals, u, MTF, XTF, Index, Radius),
        )
        prob = NonlinearProblem(fun, PX_Part)
        sol = solve(
            prob,
            NonlinearSolve.NLsolveJL(),
            abstol = abstol,
            reltol = reltol,
            maxiters = ifelse(it == 1, initial_maxiters, maxiters),
        )
        if (it > 1) && !SciMLBase.successful_retcode(sol)
            start_index = 2 + (it - 1) * (Radial_Order - 1)
            PX.WW[:, start_index:end, :, :] .= NaN
            PX.RR[start_index:end] .= NaN
            PX.TT[start_index:end] .= NaN
            break
        end
        Set_Part!(PM, PX, Intervals, sol.u)
        Tangent_Extend!(PM, PX, it)
    end
    return PM, PX
end

function Curves(
    PM::Polar_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    XX;
    Transformation = PM.Transformation,
    Damping_By_Derivative::Bool = true,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    last_index = findlast(!isnan, XX.RR)
    Radial_Mesh = PM.Radial_Mesh[1:last_index]
    WW1 = zeros(eltype(XX), State_Dimension, last_index, Skew_Dimension, Phase_Dimension)
    RR1 = zeros(eltype(XX), last_index)
    TT1 = zeros(eltype(XX), last_index)
    WW1 .= XX.WW[:, 1:last_index, :, :]
    RR1 .= XX.RR[1:last_index]
    TT1 .= XX.TT[1:last_index]
    for k in axes(WW1, 2)
        WW1[:, k, :, :] .-= PM.W0
    end
    #
    New_Grid = Radial_Mesh # range(0,1,length=100)
    MM, D_MM = Clenshaw_Interpolate(New_Grid, Radial_Order, Radial_Mesh)
    #     D_MM = MM * Barycentric_Differentiation_Matrix(Radial_Grid)
    if Damping_By_Derivative
        RR = D_MM * RR1
    else
        RR = deepcopy(RR1)
        RR[2:end] ./= New_Grid[2:end]
        RR[1] = dot(D_MM[1, :], RR1)
    end
    TT = MM * TT1
    #
    @tullio WW[i, r, k, l] := Transformation[i, k, p] * WW1[p, j, k, l] * MM[r, j]
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    AA = sqrt.(vec(sum(WW .* WW, dims = (1, 3, 4))) ./ Factor)
    return RR, TT, AA
end

# function Polar_Grid_To_Latent(MTF::Multi_Foliation, XTF, Index::Integer,
#                               PM::Polar_Manifold{State_Dimension, Latent_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension}, XX
#                              ) where {State_Dimension, Latent_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension}
#     WW = zeros(eltype(XX), State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
#     WW .= XX.WW
#     Radial_Mesh = PM.Radial_Mesh
#     Beta_Grid = Fourier_Grid(Phase_Dimension)
#     MON_LIN, D_MON_LIN = Clenshaw_Interpolate(Radial_Mesh, Radial_Order, Radial_Mesh)
#     #
#     @tullio E_WW[s, r, p, q] := WW[s, j, p, q] * MON_LIN[r, j]
#     E_WW_RS = reshape(E_WW, size(E_WW, 1), :)
#     #
#     E_Phase = zeros(Skew_Dimension, size(E_WW)[2:end]...)
#     for s in axes(E_Phase, 1), r in axes(E_Phase, 2), p in axes(E_Phase, 3), q in axes(E_Phase, 4)
#         E_Phase[s, r, p, q] = I[s, p]
#     end
#     E_Phase_RS = reshape(E_Phase, size(E_Phase, 1), :)
#     #
#     E_Input = zeros(2, size(E_WW)[2:end]...)
#     E_Input[1, :, :, :] .= reshape(Radial_Mesh, :, 1, 1)
#     E_Input[2, :, :, :] .= reshape(Beta_Grid, 1, 1, :)
#     #
#     V1_E_WW_RS = zeros(eltype(WW), Latent_Dimension, size(E_WW_RS, 2))
#     Evaluate_Encoder_First!(V1_E_WW_RS, MTF, XTF, Index, E_WW_RS, E_Phase_RS)
#     E_Output = reshape(V1_E_WW_RS, Latent_Dimension, size(E_WW)[2:end]...)
#     return E_Output, E_Input
# end

# using Natural Neighnours
# function Inverse_Interpolate(Output, Input)
#     Itp = []
#     for k in axes(Output, 3)
#         xx = vec(Input[1, :, k, :]) .* cos.(vec(Input[2, :, k, :]))
#         yy = vec(Input[1, :, k, :]) .* sin.(vec(Input[2, :, k, :]))
#         itp_X = NaturalNeighbours.interpolate(vec(Output[1, :, k, :]), vec(Output[2, :, k, :]), xx)
#         itp_Y = NaturalNeighbours.interpolate(itp_X.triangulation, yy)
#         push!(Itp, (itp_X, itp_Y))
#     end
#     return Itp
# end

# function Manifold_Coordinates(Interpolation, Latent_Data, Encoded_Phase)
#     Result = [similar(Latent_Data) for k in 1:length(Interpolation)]
#     for k in eachindex(Interpolation)
#         Result[k][1, :] .= Interpolation[k][1](Latent_Data[1, :], Latent_Data[2, :])
#         Result[k][2, :] .= Interpolation[k][2](Latent_Data[1, :], Latent_Data[2, :])
#     end
#     for k in eachindex(Result)
#         Result[k] .*= reshape(Encoded_Phase[k, :], 1, :)
#     end
#     # checking validity
#     Valid = [zeros(Bool, size(Latent_Data, 2)) for k in 1:length(Interpolation)]
#     for k in eachindex(Interpolation)
#         tri = Interpolation[k][1].triangulation
#         for j in axes(Latent_Data, 2)
#             inside = DelaunayTriangulation.find_polygon(tri, (Latent_Data[1, j], Latent_Data[2, j]))
#             Valid[k][j] = inside != 0
#         end
#     end
#     Latent_Normal = sum(Result)
#     Valid_Normal = reduce(.&, Valid)
#     return Latent_Normal, Valid_Normal
# end

# function Embed_Manifold(PM::Polar_Manifold{State_Dimension, Latent_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension}, PX,
#                         Latent_Normal, Encoded_Phase; Transformation=PM.Transformation
#                        ) where {State_Dimension, Latent_Dimension, Radial_Order, Radial_Dimension, Phase_Dimension, Skew_Dimension}
#     WW = PX.WW
#     TT = PX.TT
#     Radial_Mesh = PM.Radial_Mesh
#     Beta_Grid = Fourier_Grid(Phase_Dimension)
#     RR = sqrt.(Latent_Normal[1, :] .^ 2 + Latent_Normal[2, :] .^ 2)
#     BB = angle.(Latent_Normal[1, :] .+ 1im * Latent_Normal[2, :])
#     ITP = Barycentric_Interpolation_Matrix(Radial_Order, Radial_Mesh, RR)
# #     ITP, _ = Clenshaw_Interpolate(RR, Radial_Order, Radial_Mesh)
#     @tullio SH_beta[q, k] := psi(BB[k] - Beta_Grid[q], Phase_Dimension)
#     @tullio WW_TR[i2, j, q, l] := Transformation[i2, q, i] * WW[i, j, q, l]
#     @show size(ITP), size(SH_beta), size(Encoded_Phase)
#     @tullio E_WW[i, k] := ITP[k, r] * WW_TR[i, r, j, q] * SH_beta[q, k] * Encoded_Phase[j, k]
#     @tullio E_TT[k] := ITP[k, r] * TT[r]
#     return E_WW, E_TT
# end

function To_Latent(MTF::Multi_Foliation, XTF, Index, Data, Encoded_Phase)
    Latent_Data = zeros(eltype(Data), size(XTF.x[Index].x[1].WW, 1), size(Data, 2))
    Evaluate!(Latent_Data, MTF[Index][2], XTF.x[Index].x[2], Data, Encoded_Phase)
    return Latent_Data
end

# function Latent_To_Manifold(PM::Polar_Manifold, PX,
#                           MTF::Multi_Foliation, XTF, Index, Latent_Data, Encoded_Phase;
#                           Transformation=PM.Transformation)
#     Output, Input = Polar_Grid_To_Latent(MTF, XTF, Index, PM, PX)
#     Interpolation = Inverse_Interpolate(Output, Input)
#     Latent_Normal, Valid_Normal = Manifold_Coordinates(Interpolation, Latent_Data, Encoded_Phase)
#     E_WW, E_TT = Embed_Manifold(PM, PX, Latent_Normal, Encoded_Phase; Transformation=Transformation)
#     return E_WW, E_TT, Valid_Normal
# end

# Phase_Dimension : how many points to consider within a period of vibration
# function Trajecory_To_Amplitude(E_WW, E_TT, Valid_Normal, Index_List, Phase_Dimension::Integer)
#     E_FF = 2 * pi ./ E_TT
#     E_FF[findall(.! Valid_Normal)] .= 0
#     AMP = zero(E_TT)
#     for k in 2:length(Index_List)
#         start = 1+Index_List[k-1]
#         fin = Index_List[k]
#         shift = ceil(Int, min(maximum(E_FF[start:fin]), fin))
#         for j in start:(fin - shift)
#             if Valid_Normal[j] && (E_TT[j] > 0)
#                 res = zero(eltype(E_WW))
#                 if size(E_WW, 1) == 1
#                     itp = Interpolations.interpolate((0:shift,), E_WW[1,j:j+shift], Interpolations.Gridded(Interpolations.Linear()))
#                     for q in range(0, E_FF[j], length=Phase_Dimension+1)[1:end-1]
#                         res += itp(q) ^ 2
#                     end
#                 else
#                     itp = Interpolations.interpolate((axes(E_WW, 1), 0:shift,), E_WW[:,j:j+shift], (Interpolations.NoInterp(), Interpolations.Gridded(Interpolations.Linear())))
#                     for p in axes(E_WW, 1), q in range(0, E_FF[j], length=Phase_Dimension+1)[1:end-1]
#                         res += itp(p, q) ^ 2
#                     end
#                 end
#                 AMP[j] = sqrt(2 * res / Phase_Dimension)
#             end
#         end
#     end
#     return AMP
# end

function Error_Statistics(
    On_Manifold_Amplitude_In,
    Training_Errors,
    maxAmp,
    bins = (40, 20),
)
    minU = eps(1.0)
    Id_Valid = findall(x -> !isnan(x), On_Manifold_Amplitude_In)
    if isempty(Id_Valid)
        return eps(1.0), eps(1.0), eps(1.0), eps(1.0), eps(1.0), eps(1.0), eps(1.0)
    end
    On_Manifold_Amplitude = On_Manifold_Amplitude_In[Id_Valid]
    maxU = min(maximum(On_Manifold_Amplitude), maxAmp)
    minT = max(minimum(Training_Errors), eps(1.0))
    maxT = max(maximum(Training_Errors), eps(1.0))
    minT = ifelse(isnan(minT), eps(1.0), minT)
    maxT = ifelse(isnan(maxT), eps(1.0), maxT)
    @show minT, maxT
    edU = (
        exp.(range(log(minT), log(maxT), length = bins[1])),
        range(minU, maxU, length = bins[2]),
    )
    hsU = fit(
        Histogram{Float64},
        (Training_Errors, On_Manifold_Amplitude),
        UnitWeights{Float64}(length(Training_Errors)),
        edU,
    )
    nzinv = x -> iszero(x) ? 1 : 1 / x
    hsU.weights .*= nzinv.(sum(hsU.weights, dims = 1))

    pp = collect((hsU.edges[2][1:(end-1)] + hsU.edges[2][2:end]) / 2)
    qq = collect((hsU.edges[1][1:(end-1)] + hsU.edges[1][2:end]) / 2)
    maxidx = [findlast(hsU.weights[:, k] .> 0) for k in axes(hsU.weights, 2)]
    minidx = [findfirst(hsU.weights[:, k] .> 0) for k in axes(hsU.weights, 2)]
    maxok = findall(maxidx .!= nothing)
    minok = findall(minidx .!= nothing)
    errMaxX = qq[maxidx[maxok]]
    errMaxY = pp[maxok]
    errMinX = qq[minidx[minok]]
    errMinY = pp[minok]
    errMeanX = [mean(qq, weights(hsU.weights[:, k])) for k in axes(hsU.weights, 2)]
    errMeanY = deepcopy(pp)
    errStdX = [std(qq, weights(hsU.weights[:, k])) for k in axes(hsU.weights, 2)]
    hsU.edges[1][1] = hsU.edges[1][2]
    return errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX
end

"""
    Create_Plot()

Creates a figure to display the results.
"""
function Create_Plot()
    fig = Figure(size = (1250, 250), fontsize = 16)
    axDense = Makie.Axis(
        fig[1, 1],
        xscale = log10,
        xticks = LogTicks(WilkinsonTicks(3, k_min = 3, k_max = 4)),
    )
    axErr = Makie.Axis(
        fig[1, 2],
        xscale = log10,
        xticks = LogTicks(WilkinsonTicks(3, k_min = 3, k_max = 4)),
    )
    axTrace = Makie.Axis(
        fig[1, 3],
        yscale = log10,
        yticks = LogTicks(WilkinsonTicks(3, k_min = 3, k_max = 4)),
    )
    axFreq = Makie.Axis(fig[1, 4], xticks = WilkinsonTicks(3, k_min = 3, k_max = 4))
    axDamp = Makie.Axis(fig[1, 5], xticks = WilkinsonTicks(3, k_min = 3, k_max = 4))
    axDense.xlabel = "Data Density"
    axDense.ylabel = "Amplitude"
    axErr.xlabel = L"E_{\mathit{rel}}"
    axErr.ylabel = "Amplitude"
    axTrace.xlabel = "Iteration"
    axTrace.ylabel = "Error"
    axFreq.xlabel = "Frequency"
    axFreq.ylabel = "Amplitude"
    axDamp.xlabel = "Damping ratio"
    axDamp.ylabel = "Amplitude"
    return fig
end

"""
    Annotate_Plot!(fig)

Perform the final annotation of the figure, so that it is ready for display.
"""
function Annotate_Plot!(fig)
    axDense = content(fig[1, 1])
    axErr = content(fig[1, 2])
    axTrace = content(fig[1, 3])
    axFreq = content(fig[1, 4])
    axDamp = content(fig[1, 5])
    fig[1, 6] = Legend(
        fig,
        axFreq,
        merge = true,
        unique = true,
        labelsize = 16,
        backgroundcolor = (:white, 0),
        framevisible = false,
        rowgap = 1,
    )
    text!(axDense, "a)", space = :relative, position = Point2f(0.1, 0.9))
    text!(axErr, "b)", space = :relative, position = Point2f(0.1, 0.9))
    text!(axTrace, "c)", space = :relative, position = Point2f(0.1, 0.9))
    text!(axFreq, "d)", space = :relative, position = Point2f(0.1, 0.9))
    text!(axDamp, "e)", space = :relative, position = Point2f(0.1, 0.9))
    return fig
end

"""
    Plot_Error_Trace(
        fig,
        Index,
        Train_Trace,
        Test_Trace = nothing;
        Train_Color = Makie.wong_colors()[1],
        Test_Color = Makie.wong_colors()[2],
    )

Adds the training and testing errors to the figure `fig` for the foliation selected by `Index`.
`Train_Trace` and `Test_Trace` are produced by [`Optimise!`](@ref).
"""
function Plot_Error_Trace(
    fig,
    Index,
    Train_Trace,
    Test_Trace = nothing;
    Train_Color = Makie.wong_colors()[1],
    Test_Color = Makie.wong_colors()[2],
)
    axTrace = content(fig[1, 3])
    Id = findlast(Train_Trace[1, Index, :] .!= 0)
    lines!(axTrace, 2:Id, Train_Trace[1, Index, 2:Id], color = Train_Color)
    ymax = maximum(Train_Trace[1, Index, 2:Id])
    ymin = minimum(Train_Trace[1, Index, 2:Id])
    if !isnothing(Test_Trace)
        Id_Test = findall(Test_Trace[1, Index, :] .!= 0)
        scatter!(
            axTrace,
            Id_Test[2:end],
            Test_Trace[1, Index, Id_Test[2:end]],
            color = Test_Color,
        )
        ymax = max(maximum(x->isnan(x) ? -Inf : x, Test_Trace[1, Index, Id_Test[2:end]]), ymax)
        ymin = min(minimum(x->isnan(x) ? Inf : x, Test_Trace[1, Index, Id_Test[2:end]]), ymin)
        ylims!(axTrace, 0.9 * ymin, 1.1 * ymax)
        xlims!(axTrace, 1, Id + 1)
    else
        ylims!(axTrace, 0.9 * ymin, 1.1 * ymax)
        xlims!(axTrace, 1, Id + 1)
    end
    nothing
end

function Data_Density(data, Points_Per_Bin = 64)
    # remove zero data points. That can occur, because of incomplete manifold embedding
    # the data is sorted by magnitude
    data_nz = sort(data[findall(!iszero, data)])
    # make sure that each bin has the same lenght
    cut = Points_Per_Bin * div(length(data_nz), Points_Per_Bin)
    # create a matrix where each column is a bin
    data_nz_R = reshape(data_nz[1:cut], Points_Per_Bin, :)
    data_min = vec(minimum(data_nz_R, dims = 1))
    data_max = vec(maximum(data_nz_R, dims = 1))
    data_max[end] = maximum(data_nz)
    data_separation = (data_max[1:end-1] + data_min[2:end]) / 2
    data_max[1:end-1] .= data_separation
    data_min[2:end] .= data_separation
    data_mean = vec(mean(data_nz_R, dims = 1))
    data_tail = data_nz[(1+cut-Points_Per_Bin):end]
    data_mean[end] = mean(data_tail)
    data_points = Points_Per_Bin * ones(length(data_mean))
    data_points[end] = length(data_tail)
    #
    pos = data_mean
    den =
        data_points ./
        ((data_max .- data_min) .* (sum(data_points) * (data_max[end] - data_min[1])))
    return pos, den
end

"""
    Plot_Data_Error!(
        fig,
        PM,
        PX,
        MIP,
        XIP,
        MTF::Multi_Foliation,
        XTF,
        Index,
        Index_List,
        Data,
        Encoded_Phase;
        Transformation,
        Color = Makie.wong_colors()[1],
        Model_IC = false,
    )

Plots the training and testing errors as a function of vibration amplitude.
* `fig` the figure to plot to
* `PM`, `PX` the invariant manifold calculated by [`Find_DATA_Manifold`](@ref)
* `MIP`, `XIP` the manifold embedding calculated by [`Extract_Manifold_Embedding`](@ref)
* `MTF`, `XTF` the set of invariant foliations
* `Index` which invariant foliation is it calculated for
* `Index_List`, `Data`, `Encoded_Phase` the training data
* `Transformation` the transformation the brings back the data to the physical coordinate
* `Color` line colour of the plot
* `Model_IC` whether to re-calculate the initial conditions of trajectories stored in `XTF`
"""
function Plot_Data_Error!(
    fig,
    PM::Polar_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    PX,
    MIP,
    XIP,
    MTF::Multi_Foliation,
    XTF,
    Index,
    Index_List,
    Data,
    Encoded_Phase;
    Transformation,
    Color = Makie.wong_colors()[1],
    Model_IC = false,
    Iterations = 32,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    axDense = content(fig[1, 1])
    axErr = content(fig[1, 2])
    MTF_Cache = Make_Cache(
        MTF,
        XTF,
        Index_List,
        Data,
        Encoded_Phase,
        Model = false,
        Big_Jac = false,
        Model_IC = Model_IC,
        Iterations = Iterations,
    )
    Training_Errors = zeros(eltype(Data), size(Data, 2))
    Pointwise_Error!(
        Training_Errors,
        MTF[Index],
        XTF.x[Index],
        Index_List,
        Data,
        Encoded_Phase,
        MTF_Cache.Scaling,
        Cache = MTF_Cache.Parts[Index],
    )
    #
    Latent_Data = To_Latent(MTF, XTF, Index, Data, Encoded_Phase)
    #     E_WW, E_TT, Valid_Normal = Latent_To_Manifold(PM, PX, MTF, XTF, Index, Latent_Data, Encoded_Phase; Transformation=Transformation)
    #     Valid_Id = findall(Valid_Normal)
    #     On_Manifold_Amplitude = Trajecory_To_Amplitude(E_WW, E_TT, Valid_Normal, Index_List, Phase_Dimension)
    E_WW, E_WW_I, On_Manifold_Amplitude, Valid_Ind = Embed_Manifold(
        MIP,
        XIP,
        Latent_Data,
        Encoded_Phase;
        Output_Inverse_Transformation = Transformation,
    )
    Valid_Id = findall(Valid_Ind)
    #
    pos, den = Data_Density(On_Manifold_Amplitude[Valid_Id])
    lines!(axDense, den, pos, color = Color, linewidth = 3)

    Data_Max = maximum(pos)
    errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX = Error_Statistics(
        On_Manifold_Amplitude[Valid_Id],
        Training_Errors[Valid_Id],
        Data_Max,
    )
    lines!(axErr, errMaxX, errMaxY, color = Color, linestyle = :dot)
    lines!(axErr, errMeanX, errMeanY, color = Color, linestyle = :solid, linewidth = 3)
    return MTF_Cache, Data_Max
end

# TODO add input: On_Manifold_Data, Valid_Id
"""
    Plot_Data_Result!(
        fig,
        PM::Polar_Manifold,
        PX,
        MIP,
        XIP,
        MTF::Multi_Foliation,
        XTF,
        Index,
        Index_List,
        Data,
        Encoded_Phase;
        Data_Max = 1.0,
        Time_Step = 1.0,
        Transformation = PM.Transformation,
        Slice_Transformation = Transformation,
        Label = "Data",
        Color = Makie.wong_colors()[1],
        Hz = false,
        Damping_By_Derivative::Bool = true,
        Model_IC = false,
    )

Plots the instantaneous frequency and damping curves for the invariant foliation at `Index`.
* `fig` the figure to plot to
* `PM`, `PX` the invariant manifold calculated by [`Find_DATA_Manifold`](@ref)
* `MIP`, `XIP` the manifold embedding calculated by [`Extract_Manifold_Embedding`](@ref)
* `MTF`, `XTF` the set of invariant foliations
* `Index` which invariant foliation is it calculated for
* `Index_List`, `Data`, `Encoded_Phase` the training data
* `Data_Max` relative maximum amplitude of data to consider. `=1.0` means all data included.
* `Time_Step` sampling time-step of data for calulating frequencies
* `Transformation` the transformation the brings back the data to the physical coordinate
* `Slice_Transformation` the transformation, in case `PM`, `PX` are calculated from a fixed parameters slice of the identified foliation.
    If not a slice, should be the same as `Transformation`.
* `Label` the legend of the plotted lines
* `Color` line colour of the plot
* `Hz` if `true` frequency is displayed in Hz, otherwise it is rad/s.
* `Damping_By_Derivative` if `true` a truely instantaneous damping ratio is displayed.
    If `false` the damping ratio commonly (and mistakenly) used in the literature is displayed
* `Model_IC` whether to re-calculate the initial conditions of trajectories stored in `XTF`
"""
function Plot_Data_Result!(
    fig,
    PM::Polar_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    PX,
    MIP,
    XIP,
    MTF::Multi_Foliation,
    XTF,
    Index,
    Index_List,
    Data,
    Encoded_Phase;
    Data_Max = 1.0,
    Time_Step = 1.0,
    Transformation = PM.Transformation,
    Slice_Transformation = Transformation,
    Label = "Data",
    Color = Makie.wong_colors()[1],
    Hz = false,
    Damping_By_Derivative::Bool = true,
    Model_IC = false,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    axDense = content(fig[1, 1])
    axErr = content(fig[1, 2])
    axFreq = content(fig[1, 4])
    axDamp = content(fig[1, 5])

    RR, TT, AA = Curves(
        PM,
        PX,
        Transformation = Slice_Transformation,
        Damping_By_Derivative = Damping_By_Derivative,
    )
    if Hz
        Frequency = TT ./ (2 * pi * Time_Step)
        lines!(axFreq, Frequency, AA, label = Label, color = Color)
        axFreq.xlabel = "Frequency [Hz]"
    else
        Frequency = TT ./ Time_Step
        lines!(axFreq, Frequency, AA, label = Label, color = Color)
        axFreq.xlabel = "Frequency [rad/s]"
    end
    Damping_Ratio = -log.(abs.(RR)) ./ TT
    lines!(axDamp, Damping_Ratio, AA, label = Label, color = Color)
    MTF_Cache, Data_Max = Plot_Data_Error!(
        fig,
        PM,
        PX,
        MIP,
        XIP,
        MTF,
        XTF,
        Index,
        Index_List,
        Data,
        Encoded_Phase;
        Transformation = Transformation,
        Color = Color,
        Model_IC = Model_IC,
    )
    if Data_Max > 0
        ylims!(axDense, 0, Data_Max)
        ylims!(axErr, 0, Data_Max)
        ylims!(axFreq, 0, Data_Max)
        ylims!(axDamp, 0, Data_Max)
        #
        Plot_Id = findall(AA .<= Data_Max)
        #
        min_f = minimum(Frequency[Plot_Id])
        max_f = maximum(Frequency[Plot_Id])
        margin_f = (max_f - min_f) / 10
        xlims!(axFreq, max(0, min_f - margin_f), min(2 / Time_Step, max_f + margin_f))
        #
        min_d = minimum(Damping_Ratio[Plot_Id])
        max_d = maximum(Damping_Ratio[Plot_Id])
        margin_d = (max_d - min_d) / 10
        xlims!(axDamp, max(0, min_d - margin_d), min(1, max_d + margin_d))
    end
    return MTF_Cache
end

"""
    Plot_Model_Result!(
        fig,
        PM::Polar_Manifold,
        PX;
        Time_Step = 1.0,
        Label = "MAP Model",
        Color = Makie.wong_colors()[2],
        Hz = false,
        Damping_By_Derivative::Bool = true,
    )

Display the instantaneous frequency and damping ratio for the invariant manifold directly calculated from an ODE or map model.
* `fig` the figure to plot to
* `PM`, `PX` the invariant manifold calculated by [`Find_ODE_Manifold`](@ref) or [`Find_MAP_Manifold`](@ref)
* `Time_Step` sampling time-step of data for calulating frequencies
* `Color` line colour of the plot
* `Hz` if `true` frequency is displayed in Hz, otherwise it is rad/s.
* `Damping_By_Derivative` if `true` a truely instantaneous damping ratio is displayed.
    If `false` the damping ratio commonly (and mistakenly) used in the literature is displayed
"""
function Plot_Model_Result!(
    fig,
    PM::Polar_Manifold{
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    },
    PX;
    Time_Step = 1.0,
    Label = "MAP Model",
    Color = Makie.wong_colors()[2],
    Hz = false,
    Damping_By_Derivative::Bool = true,
) where {
    State_Dimension,
    Latent_Dimension,
    Radial_Order,
    Radial_Dimension,
    Phase_Dimension,
    Skew_Dimension,
}
    #
    axDense = content(fig[1, 1])
    axErr = content(fig[1, 2])
    axFreq = content(fig[1, 4])
    axDamp = content(fig[1, 5])
    #
    RR, TT, AA = Curves(PM, PX; Damping_By_Derivative = Damping_By_Derivative)
    if Hz
        lines!(axFreq, TT ./ (2 * pi * Time_Step), AA, label = Label, color = Color)
    else
        lines!(axFreq, TT ./ Time_Step, AA, label = Label, color = Color)
    end
    lines!(axDamp, -log.(abs.(RR)) ./ TT, AA, label = Label, color = Color)
    return fig
end
