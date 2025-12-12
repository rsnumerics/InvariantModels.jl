# SPDX-License-Identifier: EUPL-1.2

# invariance equation
# D1 W . R + omega * D2 W + D3 W . T = F ( W )

struct Polar_ODE_Manifold{
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
    M_Model::MultiStep_Model
    X_Model::Any
    Generator::Any
    W0::Any   # 4 dimensional array : n_s x n_r x n_b x n_t
    R0::Any   # 1 dimensional array : n_r  (polynomial coefficients)
    T0::Any   # 1 dimensional array : n_r  (polynomial coefficients)
end

function Take_Part(
        PP::Polar_ODE_Manifold{
            State_Dimension,
            Latent_Dimension,
            Radial_Order,
            Radial_Dimension,
            Phase_Dimension,
            Skew_Dimension,
        },
        XX_Full,
        Intervals::Tuple{Integer, Integer},
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
        PP::Polar_ODE_Manifold{
            State_Dimension,
            Latent_Dimension,
            Radial_Order,
            Radial_Dimension,
            Phase_Dimension,
            Skew_Dimension,
        },
        XX_Full,
        Intervals::Tuple{Integer, Integer},
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
        PP::Polar_ODE_Manifold{
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
    Scaling = Radial_Mesh[(final_index + 1):end] .- Radial_Mesh[final_index]
    RR[(final_index + 1):end] .= RR[final_index] .+ D_RR * Scaling
    TT[(final_index + 1):end] .= TT[final_index] .+ D_TT * Scaling
    WW[:, (final_index + 1):end, :, :] .=
        WW[:, [final_index], :, :] .+ D_WW_R .* reshape(Scaling, 1, :, 1, 1)
    return nothing
end

function Polar_ODE_Manifold(
        M_Model::MultiStep_Model{
            State_Dimension,
            Skew_Dimension,
            Start_Order,
            End_Order,
            Trajectories,
        },
        X_Model,
        Generator,
        Select,
        Radial_Order,
        Radial_Intervals,
        Phase_Dimension,
        Radius,
    ) where {State_Dimension, Skew_Dimension, Start_Order, End_Order, Trajectories}
    Radial_Mesh = Chebyshev_Mesh(Radial_Order, Radial_Intervals)
    Radial_Dimension = length(Radial_Mesh)
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    # output
    WW = zeros(State_Dimension, Radial_Dimension, Skew_Dimension, Phase_Dimension)
    RR = zeros(Radial_Dimension)
    TT = zeros(Radial_Dimension)

    # Torus and decomposed linear dynamics
    Torus, Jac = Find_Torus_ODE(M_Model, X_Model, Generator)
    W_R, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values =
        Decompose_Model_ODE(Jac, Generator)
    # calculating tangent
    Wr = W_R[:, :, Select[1]]
    Wi = W_R[:, :, Select[2]]
    DR0 = real_values[Select[1]]
    T0 = real_values[Select[2]]
    #     println("Polar_ODE_Manifold: real values")
    #     @show real_values
    #     @show cplx_values
    @tullio FT[i, j, k] := Wr[i, j] * cos(Beta_Grid[k]) - Wi[i, j] * sin(Beta_Grid[k])
    #
    for k in axes(WW, 4), l in axes(WW, 2)
        #         @show size(WW[:, 1, :, k]), size(Torus)
        WW[:, l, :, k] .= Torus
    end
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    #     @show size(WW[:, 2, :, :]), size(FT)
    Amplitude = sqrt(sum(FT .* FT) / Factor)
    @show Amplitude, sqrt(sum(FT .* FT) / Factor)
    for k in eachindex(Radial_Mesh)
        WW[:, k, :, :] .+= FT .* (Radial_Mesh[k] * Radius / Amplitude)
    end
    #
    RR .= DR0 .* Radial_Mesh * Radius
    TT .= T0
    Latent_Dimension = size(X_Model.WW, 1)
    PM = Polar_ODE_Manifold{
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
        Generator,
        deepcopy(WW[:, 1, :, :]),
        RR[1],
        TT[1],
    )
    XM = ComponentArray(WW = WW, RR = RR, TT = TT)
    return PM, XM
end

function Evaluate!(
        Value,
        Jacobian,
        PP::Polar_ODE_Manifold{
            State_Dimension,
            Latent_Dimension,
            Radial_Order,
            Radial_Dimension,
            Phase_Dimension,
            Skew_Dimension,
        },
        XX_Full,
        Intervals::Tuple{Integer, Integer},
        XX_Part,
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
    E_RR = XX_Part.RR
    E_TT = XX_Part.TT
    start_index = 2 + (Intervals[1] - 1) * (Radial_Order - 1)
    final_index = 1 + Intervals[2] * (Radial_Order - 1)
    #     println("From = ", start_index, " To = ", final_index)
    WW[:, start_index:final_index, :, :] .= XX_Part.WW
    RR[start_index:final_index] .= XX_Part.RR
    TT[start_index:final_index] .= XX_Part.TT
    #
    Generator = PP.Generator
    M_Model, X_Model = PP.M_Model, PP.X_Model

    Radial_Mesh = PP.Radial_Mesh
    MON_LIN, D_MON_LIN = Clenshaw_Interpolate(
        Radial_Mesh[start_index:final_index],
        Radial_Order,
        Radial_Mesh,
    )
    grid_r = Radial_Mesh[start_index:final_index] # disregard the first collocation point
    Beta_Grid = Fourier_Grid(Phase_Dimension)
    DD_Beta = Fourier_Differential(Phase_Dimension)
    #
    # F o W - D_R W .* RR - D_Beta W .* TT - omega *  D_Theta W
    #
    @tullio E_WW[s, r, p, q] := WW[s, j, p, q] * MON_LIN[r, j]
    E_WW_RS = reshape(E_WW, size(E_WW, 1), :)
    #
    E_Phase = zeros(Skew_Dimension, size(E_WW)[2:end]...)
    for s in axes(E_Phase, 1),
            r in axes(E_Phase, 2),
            p in axes(E_Phase, 3),
            q in axes(E_Phase, 4)

        E_Phase[s, r, p, q] = I[s, p]
    end
    E_Phase_RS = reshape(E_Phase, size(E_Phase, 1), :)
    #
    WWN = deepcopy(WW)
    for k in axes(WWN, 2)
        WWN[:, k, :, :] .-= WW[:, 1, :, :]
    end
    #
    V1_R = range(1, Latent_Dimension * size(E_WW_RS, 2))
    RR_R = range(1 + last(V1_R), last(V1_R) + length(XX_Part.RR))
    TT_R = range(1 + last(RR_R), last(RR_R) + length(XX_Part.TT))
    #
    if !isnothing(Value)
        Value .= 0
        #
        F_WW_RS = zero(E_WW_RS)
        Evaluate_Function!(F_WW_RS, M_Model, X_Model, E_WW_RS, E_Phase_RS)
        F_WW = reshape(F_WW_RS, :, size(E_WW)[2:end]...)
        # the derivatives
        @tullio DR_E_WW_RR[s, r, p, q] := WW[s, j, p, q] * D_MON_LIN[r, j] * E_RR[r]
        @tullio DT_E_WW_OM[s, r, p, q] := WW[s, j, m, q] * MON_LIN[r, j] * Generator[p, m]
        @tullio DB_E_WW_TT[s, r, p, q] :=
            WW[s, j, p, m] * MON_LIN[r, j] * DD_Beta[q, m] * E_TT[r]
        # Phase conditions
        @tullio D0WW[i, r, k, l] := WWN[i, j, k, l] * MON_LIN[r, j]
        @tullio D2WW[i, r, k, l] := WWN[i, j, k, p] * DD_Beta[p, l] * MON_LIN[r, j]
        @tullio D1WW[i, r, k, l] := WWN[i, j, k, l] * D_MON_LIN[r, j]
        Factor = (Phase_Dimension * Skew_Dimension / 2)
        # entering values...
        Value[V1_R] .= vec(
            (F_WW .- DR_E_WW_RR .- DT_E_WW_OM .- DB_E_WW_TT) ./ reshape(grid_r, 1, :, 1, 1),
        )
        if !Arclength
            Value[RR_R] .=
                vec(sum(D0WW .* D1WW, dims = (1, 3, 4))) ./ grid_r .- (Radius^2 * Factor)
        else
            Value[RR_R] .= vec(sum(D1WW .* D1WW, dims = (1, 3, 4))) .- (Radius^2 * Factor)
        end
        Value[TT_R] .= vec(sum(D1WW .* D2WW, dims = (1, 3, 4))) ./ grid_r
        if eltype(Value) <: Float64
            print(" E=", norm(Value), " E1=", maximum(abs.(Value[V1_R])))
            println(
                " E_R=",
                maximum(abs.(Value[RR_R])),
                " E_T=",
                maximum(abs.(Value[TT_R])),
            )
        end
    end

    ### JACOBIAN ###
    if !isnothing(Jacobian)
        Jacobian .= 0
        print(" J")
        MON_LIN_R = view(MON_LIN, :, start_index:final_index)
        D_MON_LIN_R = view(D_MON_LIN, :, start_index:final_index)
        #
        J_F_WW_RS = zeros(eltype(WW), Latent_Dimension, State_Dimension, size(E_WW_RS, 2))
        Jacobian_Function!(J_F_WW_RS, M_Model, X_Model, E_WW_RS, E_Phase_RS)
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
        # 1.a.F
        J_F = reshape(J_F_WW_RS, :, State_Dimension, size(E_WW)[2:end]...)
        for q in axes(J_F, 5),
                p in axes(J_F, 4),
                j in axes(MON_LIN_R, 2),
                r in axes(J_F, 3),
                t in axes(J_F, 2),
                s in axes(J_F, 1)

            Jac_V1_WW[s, r, p, q, t, j, p, q] = J_F[s, t, r, p, q] * MON_LIN_R[r, j]
        end
        for q in axes(E_WW, 4),
                p in axes(E_WW, 3),
                j in axes(D_MON_LIN_R, 2),
                r in axes(E_WW, 2),
                s in axes(E_WW, 1)

            Jac_V1_WW[s, r, p, q, s, j, p, q] -= D_MON_LIN_R[r, j] * E_RR[r]
        end
        for m in axes(Generator, 2),
                q in axes(E_WW, 4),
                p in axes(E_WW, 3),
                j in axes(MON_LIN_R, 2),
                r in axes(E_WW, 2),
                s in axes(E_WW, 1)

            Jac_V1_WW[s, r, p, q, s, j, m, q] -= Generator[p, m] * MON_LIN_R[r, j]
        end
        for m in axes(DD_Beta, 2),
                q in axes(E_WW, 4),
                p in axes(E_WW, 3),
                j in axes(MON_LIN_R, 2),
                r in axes(E_WW, 2),
                s in axes(E_WW, 1)

            Jac_V1_WW[s, r, p, q, s, j, p, m] -= DD_Beta[q, m] * MON_LIN_R[r, j] * E_TT[r]
        end
        #
        # 1.b. D_RR : SH o V1 o W o (R, T)
        Id_R = Diagonal(I, size(MON_LIN, 1))
        Jac_V1_RR = reshape(
            view(Jacobian, V1_R, RR_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            length(XX_Part.RR),
        )
        @tullio Jac_V1_RR[s, r, p, q, t] = -WW[s, j, p, q] * D_MON_LIN[r, j] * Id_R[r, t]
        # 1.c. D_TT : SH o V1 o W o (R, T)
        Jac_V1_TT = reshape(
            view(Jacobian, V1_R, TT_C),
            Latent_Dimension,
            size(E_WW)[2:end]...,
            length(XX_Part.RR),
        )
        @tullio Jac_V1_TT[s, r, p, q, r] =
            -WW[s, j, p, m] * MON_LIN[r, j] * DD_Beta[q, m] * Id_R[r, t]
        # rescale
        Jac_V1 = reshape(view(Jacobian, V1_R, :), Latent_Dimension, size(E_WW)[2:end]..., :)
        Jac_V1 ./= reshape(grid_r, 1, :, 1, 1, 1)
        # 3. D0WW .* D1WW
        Jac_RR_WW =
            reshape(view(Jacobian, RR_R, WW_C), length(XX_Part.RR), size(XX_Part.WW)...)
        Jac_TT_WW =
            reshape(view(Jacobian, TT_R, WW_C), length(XX_Part.TT), size(XX_Part.WW)...)

        if !Arclength
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] :=
                MON_LIN_R[r, b] * WWN[a, j2, d, g] * D_MON_LIN[r, j2]
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] +=
                WWN[a, j1, d, g] * MON_LIN[r, j1] * D_MON_LIN_R[r, b]
            Jac_RR_WW_tmp ./= reshape(grid_r, :, 1, 1, 1, 1)
        else
            # WITH DW * DW
            @tullio Jac_RR_WW_tmp[r, a, b, d, g] :=
                2 * D_MON_LIN_R[r, b] * WWN[a, j2, d, g] * D_MON_LIN[r, j2]
        end
        Jac_RR_WW .= Jac_RR_WW_tmp
        #
        @tullio Jac_TT_WW_tmp[r, a, b, d, g] :=
            DD_Beta[g, l] * MON_LIN_R[r, b] * WWN[a, j2, d, l] * D_MON_LIN[r, j2]
        @tullio Jac_TT_WW_tmp[r, a, b, d, g] +=
            WWN[a, j1, d, p] * DD_Beta[p, g] * MON_LIN[r, j1] * D_MON_LIN_R[r, b]
        Jac_TT_WW .= Jac_TT_WW_tmp
        Jac_TT_WW ./= reshape(grid_r, :, 1, 1, 1, 1)
        if any(isnan.(Jacobian))
            print(" JNaN")
        end
    end
    return nothing
end

"""
    Find_ODE_Manifold(
        MM::MultiStep_Model,
        MX,
        Generator,
        Select;
        Radial_Order,
        Radial_Intervals,
        Radius,
        Phase_Dimension,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 12,
        initial_maxiters = 200,
    )

Calculates a two-dimensional invariant manifold from the ODE `MM`, `MX`.
The result is stored as a piecewise Chebyshev polynomial in the radial direction Fourier collocation in the angular direction.

The input arguments are
* `MM` and `MX` represent the polynomial vector field
* `Generator` is the infinitesimal generator matrix of the forcing dynamics
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
function Find_ODE_Manifold(
        MM::MultiStep_Model{State_Dimension, Skew_Dimension, Start_Order, End_Order, Trajectories},
        MX,
        Generator,
        Select;
        Radial_Order,
        Radial_Intervals,
        Radius,
        Phase_Dimension,
        abstol = 1.0e-9,
        reltol = 1.0e-9,
        maxiters = 12,
        initial_maxiters = 200,
    ) where {State_Dimension, Skew_Dimension, Start_Order, End_Order, Trajectories}
    MP, XP = Polar_ODE_Manifold(
        MM,
        MX,
        Generator,
        Select,
        Radial_Order,
        Radial_Intervals,
        Phase_Dimension,
        Radius,
    )

    for it in 1:Radial_Intervals
        println(
            "ODE: Interval = ",
            it,
            " of ",
            Radial_Intervals,
            " = ",
            it / Radial_Intervals,
        )
        Intervals = (it, it)
        XP_Part = Take_Part(MP, XP, Intervals)
        fun = NonlinearFunction(
            (res, u, p) -> Evaluate!(res, nothing, MP, XP, Intervals, u, Radius);
            jac = (J, u, p) -> Evaluate!(nothing, J, MP, XP, Intervals, u, Radius),
        )
        prob = NonlinearProblem(fun, XP_Part)
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
        Set_Part!(MP, XP, Intervals, sol.u)
        Tangent_Extend!(MP, XP, it)
    end
    return MP, XP
end

function Curves(
        PM::Polar_ODE_Manifold{
            State_Dimension,
            Latent_Dimension,
            Radial_Order,
            Radial_Dimension,
            Phase_Dimension,
            Skew_Dimension,
        },
        XX
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
    New_Grid = Radial_Mesh
    MM, D_MM = Clenshaw_Interpolate(New_Grid, Radial_Order, Radial_Mesh)
    RR = D_MM * RR1
    TT = MM * TT1
    #
    @tullio WW[i, r, k, l] := WW1[i, j, k, l] * MM[r, j]
    Factor = (Phase_Dimension * Skew_Dimension / 2)
    AA = sqrt.(vec(sum(WW .* WW, dims = (1, 3, 4))) ./ Factor)
    return RR, TT, AA
end

function Model_Result(
        PM::Polar_ODE_Manifold{
            State_Dimension,
            Latent_Dimension,
            Radial_Order,
            Radial_Dimension,
            Phase_Dimension,
            Skew_Dimension,
        },
        PX;
        Hz = false,
    ) where {
        State_Dimension,
        Latent_Dimension,
        Radial_Order,
        Radial_Dimension,
        Phase_Dimension,
        Skew_Dimension,
    }
    #
    #     axDense = content(fig[1, 1])
    #     axErr = content(fig[1, 2])
    #     axFreq = content(fig[1, 4])
    #     axDamp = content(fig[1, 5])
    #
    RR, TT, AA = Curves(PM, PX)
    if Hz
        Frequency = TT ./ (2 * pi)
        #         lines!(axFreq, Frequency, AA, label = Label, color = Color)
    else
        Frequency = TT
        #         lines!(axFreq, Frequency, AA, label = Label, color = Color)
    end
    Damping_Ratio = -RR ./ TT
    #     lines!(axDamp, -RR ./ TT, AA, label = Label, color = Color)
    Backbone_Curves = (Amplitude = AA, Frequency = Frequency, Damping_Ratio = Damping_Ratio, Hz = Hz)
    return Backbone_Curves
end
