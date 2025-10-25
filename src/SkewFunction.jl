struct SkewFunction{m,k,𝔽} <: AbstractDecoratorManifold{𝔽}
    manifold::Euclidean
end

function SkewFunction(m::Int, k::Int; field::AbstractNumbers=ℝ)
    manifold = Euclidean(m, k, field=field)
    return SkewFunction{m,k,field}(manifold)
end

ManifoldsBase.decorated_manifold(M::SkewFunction) = M.manifold
ManifoldsBase.active_traits(f, ::SkewFunction, args...) = ManifoldsBase.IsExplicitDecorator()

function Base.zero(M::Euclidean{T,𝔽}) where {T,𝔽}
    if 𝔽 == ℂ
        return zeros(ComplexF64, representation_size(M)...)
    else
        return zeros(representation_size(M)...)
    end
end

function Base.zero(M::SkewFunction)
    return zero(M.manifold)
end

function Slice(M::SkewFunction{m,k,𝔽}, X, Encoded_Slice) where {m,k,𝔽}
    M_Slice = SkewFunction(m, 1; field=𝔽)
    X_Slice = zero(M_Slice)
    X_Slice_View = view(X_Slice, :, 1)
    @tullio X_Slice_View[i] = X[i, j] * Encoded_Slice[j]
    return M_Slice, X_Slice
end

function Make_Cache(M::SkewFunction, X, Data...)
    Phase = Data[1]
    #     @show size(X), size(Phase)
    @tullio Result[j, k] := X[j, p] * Phase[p, k]
    return Result
end

function Update_Cache!(Cache, M::SkewFunction, X, Data...)
    Phase = Data[1]
    @tullio Cache[j, k] = X[j, p] * Phase[p, k]
    return nothing
end

function Evaluate!(Result, M::SkewFunction, X, Data...; Cache=Make_Cache(M, X, Data...))
    Result .= Cache
    return nothing
end

function Evaluate_Add!(Result, M::SkewFunction, X, Data...; Cache=Make_Cache(M, X, Data...))
    Result .+= Cache
    return nothing
end

function L0_DF!(
    DF, M::SkewFunction, X, Data::Vararg{AbstractMatrix{T}}; L0, Cache=Make_Cache(M, X, Data...)) where {T}
    Phase = Data[1]
    #     @show size(Phase)
    @tullio DF[i, q] = L0[i, k] * Phase[q, k]
    return nothing
end

function Scaled_Hessian!(
    HH, M::SkewFunction, X, Data::Vararg{AbstractMatrix{T}}; Scaling, Cache=Make_Cache(M, X, Data...)) where {T}
    Phase = Data[1]
    Id = Diagonal(I, size(X, 1))
    @show size(HH), size(Id)
    @tullio HH[i1, q1, i2, q2] = Id[i1, i2] * Scaling[k] * Phase[q1, k] * Phase[q2, k]
    return nothing
end

# Calculate HH[i1, q1, i2, q2] = L0[i1, i2, k] * Phase[q1, k] * Phase[q2, k]
function L0_DF_DF_Delta!(DF, Delta, Latent_Delta, M::SkewFunction, X, Data::Vararg{AbstractMatrix{T}};
    Scaling, Cache=Make_Cache(M, X, Data...)) where {T}
    Phase = Data[1]
    @tullio DF[i2, q2] = Scaling[k] * Phase[q1, k] * Delta[i2, q1] * Phase[q2, k]
    return nothing
end

@inline riemannian_Hessian!(M::SkewFunction, Y, p, G, H, X) = riemannian_Hessian!(M.manifold, Y, p, G, H, X)

function Jacobian_Add!(Jac, M::SkewFunction, X, Data...; Cache=nothing)
    nothing
end
