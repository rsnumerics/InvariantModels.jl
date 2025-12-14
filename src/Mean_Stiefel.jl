# SPDX-License-Identifier: EUPL-1.2

struct Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽} <: AbstractManifold{𝔽}
end

function Mean_Flat_Stiefel(Rows::Int, Columns::Int, Skew_Dimension::Int; field::AbstractNumbers = ℝ)
    return Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, field}()
end

function ManifoldsBase.manifold_dimension(
        M::Mean_Flat_Stiefel{
            Rows, Columns, Skew_Dimension, ℝ,
        }
    ) where {Rows, Columns, Skew_Dimension}
    return Columns * Skew_Dimension * Rows - div(Rows * (Rows + 1), 2)
end

function ManifoldsBase.manifold_dimension(
        M::Mean_Flat_Stiefel{
            Rows, Columns, Skew_Dimension, ℂ,
        }
    ) where {Rows, Columns, Skew_Dimension}
    return 2 * Columns * Skew_Dimension * Rows - Rows * Rows
end

function ManifoldsBase.inner(M::Mean_Flat_Stiefel, p, X, Y)
    return dot(X, Y)
end

@inline ManifoldsBase.default_retraction_method(::Mean_Flat_Stiefel) = PolarRetraction()
@inline ManifoldsBase.default_inverse_retraction_method(::Mean_Flat_Stiefel) = PolarInverseRetraction()

function ManifoldsBase.representation_size(
        M::Mean_Flat_Stiefel{
            Rows, Columns, Skew_Dimension, 𝔽,
        }
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    #     println("representation_size")
    return (Rows, Columns, Skew_Dimension)
end

function Base.zero(
        M::Mean_Flat_Stiefel{
            Rows, Columns, Skew_Dimension, field,
        }
    ) where {Rows, Columns, Skew_Dimension, field}
    return zeros(Rows, Columns, Skew_Dimension)
end

function ManifoldsBase.zero_vector!(
        M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, field}, X, p
    ) where {Rows, Columns, Skew_Dimension, field}
    X .= 0
    return X
end

# CORRECT
function Manifolds.project!(
        M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, field}, q, p
    ) where {Rows, Columns, Skew_Dimension, field}
    p_R = reshape(p, size(p, 1), :)
    q_R = reshape(q, size(q, 1), :)
    #
    s = svd(p_R)
    mul!(q_R, s.U, s.Vt, sqrt(Skew_Dimension), 0)
    println("project! - 3 arg")
    if any(isnan.(q_R))
        println("NaN Encountered")
    end
    return q # reshape(q_R, size(q)...)
end

# CORRECT
function Manifolds.project!(
        ::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, field}, Y, p, X
    ) where {Rows, Columns, Skew_Dimension, field}
    p_R = reshape(p, size(p, 1), :)
    X_R = reshape(X, size(X, 1), :)
    Y_R = reshape(Y, size(Y, 1), :)
    #
    A = p_R * X_R'
    copyto!(Y_R, X_R - Hermitian((A + A') / (2 * Skew_Dimension)) * p_R)
    #     println("project! - 4 arg")
    #     @show norm(Y)
    #     if any(isnan.(Y))
    #         println("NaN Encountered")
    #     end
    return Y # reshape(Y_R, size(Y)...)
end

function ManifoldsBase.retract_polar!(M::Mean_Flat_Stiefel, q, p, X)
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, one(eltype(p)))
end
# CORRECT
function ManifoldsBase.retract_polar_fused!(
        ::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, q,
        p, X, t::Number
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    #     println("retract_polar_fused!")
    #     if any(isnan.(q))
    #         println("q NaN Encountered")
    #     end
    #     if any(isnan.(p))
    #         println("p NaN Encountered")
    #     end
    q .= p .+ t .* X
    q_R = reshape(q, size(q, 1), :)
    #     display(q_R)
    s = svd(q_R)
    mul!(q_R, s.U, s.Vt, sqrt(Skew_Dimension), 0)
    return q # reshape(q_R, size(q)...)
end

# inverse_retract(::Stiefel, ::Any, ::Any, ::PolarInverseRetraction)
# function inverse_retract_polar!(::Mean_Flat_Stiefel, X, p, q)
#     X_R = reshape(X, size(X, 1), :)
#     p_R = reshape(p, size(p, 1), :)
#     q_R = reshape(q, size(q, 1), :)
#     println("WRONG INVERSE RETRACTION")
#     A = p_R * q_R'
#     H = -2 * one(p * p')
#     B = lyap(A, H)
#     mul!(X_R, B, q_R)
#     X_R .-= p_R
#     println("inverse_retract_polar!")
#     if any(isnan.(X_R))
#         println("NaN Encountered")
#     end
#     return X # reshape(X_R, size(X)...)
# end

# function vector_transport_direction_diff!(M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, Y, p, X, d, ::PolarRetraction) where {Rows, Columns, Skew_Dimension, 𝔽}
#     q = retract(M, p, d, PolarRetraction())
#     d_R = reshape(d, size(d, 1), :)
#     q_R = reshape(q, size(q, 1), :)
#     X_R = reshape(X, size(X, 1), :)
#     Y_R = reshape(Y, size(Y, 1), :)
#     Iddsqrt = sqrt(I + d_R * d_R')
#     Λ = sylvester(Iddsqrt, Iddsqrt, -q_R * X_R' + X_R * q_R')
#     copyto!(Y_R, Λ * q_R  + (X_R - (q_R * X_R') * q_R) / Iddsqrt)
#     println("vector_transport_direction_diff!")
#     if any(isnan.(Y_R))
#         println("NaN Encountered")
#     end
#     return Y # reshape(Y_R, size(Y)...)
# end

# function vector_transport_to_diff!(M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, Y, p, X, q, ::PolarRetraction) where {Rows, Columns, Skew_Dimension, 𝔽}
#     d = inverse_retract(M, p, q, PolarInverseRetraction())
#     d_R = reshape(d, size(d, 1), :)
#     q_R = reshape(q, size(q, 1), :)
#     X_R = reshape(X, size(X, 1), :)
#     Y_R = reshape(Y, size(Y, 1), :)
#     #
#     Iddsqrt = sqrt(I + d_R * d_R')
#     Λ = sylvester(Iddsqrt, Iddsqrt, -q_R * X_R' + X_R * q_R')
#     copyto!(Y_R, Λ * q_R  + (X_R - (q_R * X_R') * q_R) / Iddsqrt)
#     println("vector_transport_to_diff!")
#     if any(isnan.(Y_R))
#         println("NaN Encountered")
#     end
#     return Y # reshape(Y_R, size(Y)...)
# end

function Make_Cache(
        M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data...
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show size(X), size(Phase)
    @tullio Result_Matrix[j, p, k] := X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] := Result_Matrix[j, p, k] * Data_State[p, k]
    return (Result_Matrix, Result_Value)
end

function Update_Cache!(
        Cache, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽},
        X, Data...
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    Result_Matrix, Result_Value = Cache
    @tullio Result_Matrix[j, p, k] = X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] = Result_Matrix[j, p, k] * Data_State[p, k]
    return nothing
end

function Evaluate!(
        Result, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Result .= Cache[2]
    return nothing
end

function Evaluate_Add!(
        Result, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Result .+= Cache[2]
    return nothing
end

function L0_DF!(
        DF, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data::Vararg{AbstractMatrix{T}};
        L0, Cache = Make_Cache(M, X, Data...)
    ) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[i, p, q] = L0[i, k] * Data_State[p, k] * Phase[q, k]
    #     println("L0_DF!")
    #     @show norm(DF)
    return nothing
end

# function Scaled_Hessian!(HH, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data::Vararg{AbstractMatrix{T}}; Scaling, Cache = nothing) where {Rows, Columns, Skew_Dimension,𝔽, T}
#     Phase = Data[1]
#     Data_State = Data[3]
#     Id = Diagonal(I, size(X, 2))
#     @tullio HH[i1, p1, q1, i2, p2, q2] = Id[i1, i2] * Scaling[k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k]
#     println("Scaled_Hessian!")
#     @show norm(HH)
#     return nothing
# end

# CALCULATE HH[p1, i1, q1, p2, i2, q2] = L0[i1, i2, k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k]
function L0_DF_DF_Delta!(
        DF, Delta, Latent_Delta, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X,
        Data::Vararg{AbstractMatrix{T}}; Scaling, Cache = nothing
    ) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]

    @tullio Latent_Delta[l, k] = Delta[l, p1, q1] * Data_State[p1, k] * Phase[q1, k]
    Latent_Delta .*= reshape(Scaling, 1, :)
    @tullio DF[l, p2, q2] = Latent_Delta[l, k] * Data_State[p2, k] * Phase[q2, k]
    #     println("L0_DF_DF_Delta!")
    #     @show norm(Delta), norm(Latent_Delta), norm(Data_State), norm(Phase)
    #     @show norm(DF)
    return nothing
end

# function Weingarten!(::Stiefel, Y, p, X, V)
#     Z = symmetrize(X' * V)
#     Y .= -X * p' * V .- p * Z
#     return Y
# end
# everything is transpose
#     Y .= (-X' * p * V' .- p' * Z).T
#     Z = symmetrize(X * V')
#     Y .= (-V * p' * X .- Z' * p
#     ZT = symmetrize(V * X')
#     Y .= (-V * p' * X .- ZT * p)

function ManifoldsBase.Weingarten!(
        ::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, Y, p, X, V
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Y_R = reshape(Y, size(Y, 1), :)
    p_R = reshape(p, size(p, 1), :)
    X_R = reshape(X, size(X, 1), :)
    V_R = reshape(V, size(V, 1), :)
    ZT = Manifolds.symmetrize(V_R * X_R')
    Y_R .= (.-V_R * p_R' * X_R .- ZT * p_R) ./ Skew_Dimension
    #     println("WEINGARTEN")
    #     @show norm(Y_R)
    #     if any(isnan.(Y_R))
    #         @show norm(ZT),norm(Y_R)
    #         println("NaN Encountered")
    #     end
    return Y # reshape(Y_R, size(Y)...)
end

function riemannian_Hessian!(M::Mean_Flat_Stiefel, Y, p, eG, eH, X)
    #     @show norm(eG), norm(eH), norm(X)
    project!(M, Y, p, eH) #first term - project the Euclidean Hessian
    #     @show norm(Y)
    Y .+= Weingarten(M, p, X, eG - project(M, p, eG))
    #     println("riemannian_Hessian!")
    #     if any(isnan.(Y))
    #         println("NaN Encountered")
    #     end
    #     @show norm(Y)
    return Y
end

function Jacobian_Add!(
        Jac, M::Mean_Flat_Stiefel{Rows, Columns, Skew_Dimension, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)
    ) where {Rows, Columns, Skew_Dimension, 𝔽}
    Jac .+= Cache[1]
    return nothing
end
