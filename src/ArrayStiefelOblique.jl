# make it into: 1st element Stiefel 2nd -- last : ArrayStiefelOblique

struct ArrayStiefelOblique{Rows, Columns, Skew_Dimension, tall, 𝔽} <: AbstractDecoratorManifold{𝔽}
    manifold::ProductManifold
end

function ArrayStiefelOblique(Rows::Int, Columns::Int, Skew_Dimension::Int; field::AbstractNumbers = ℝ, tall = true)
    if tall
        #         manifold = ProductManifold(Stiefel(Rows, Columns, field), PowerManifold(Oblique(Rows, Columns, field), Skew_Dimension-1))
        manifold = ProductManifold(
            Stiefel(Rows, Columns, field), PowerManifold(Euclidean(Rows, Columns, field = field), Skew_Dimension - 1))
    else
        #         manifold = ProductManifold(Stiefel(Columns, Rows, field), PowerManifold(Oblique(Columns, Rows, field), Skew_Dimension-1))
        manifold = ProductManifold(
            Stiefel(Columns, Rows, field), PowerManifold(Euclidean(Columns, Rows, field = field), Skew_Dimension - 1))
    end
    return ArrayStiefelOblique{Rows, Columns, Skew_Dimension, tall, field}(manifold)
end

@inline ManifoldsBase.decorated_manifold(M::ArrayStiefelOblique) = M.manifold
@inline ManifoldsBase.get_forwarding_type(::ArrayStiefelOblique, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::ArrayStiefelOblique, ::Any, _) = ManifoldsBase.SimpleForwardingType()

@inline ManifoldsBase.default_retraction_method(M::ArrayStiefelOblique) = default_retraction_method(M.manifold)

function Base.zero(M::ArrayStiefelOblique{
        Rows, Columns, Skew_Dimension, tall, field}) where {Rows, Columns, Skew_Dimension, tall, field}
    if tall
        return zeros(Rows, Columns, Skew_Dimension)
    else
        return zeros(Columns, Rows, Skew_Dimension)
    end
end

function ManifoldsBase.submanifold_components(X::Array{T, 3}) where {T}
    return (view(X, :, :, 1), view(X, :, :, 2:size(X, 3)))
end

# function ManifoldsBase.submanifold_components(M::ArrayStiefelOblique, X)
#     return (view(X, :, :, 1), view(X, :, :, 2:size(X, 3)))
# end

function ManifoldsBase.submanifold_component(X::Array{T, 3}, i) where {T}
    if i == 1
        return view(X, :, :, 1)
    elseif i == 2
        return view(X, :, :, 2:size(X, 3))
    else
        println(" submanifold_component: Error")
    end
end

function Make_Cache(M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, true, 𝔽},
        X, Data...) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show size(X), size(Phase)
    @tullio Result_Matrix[j, p, k] := X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] := Result_Matrix[j, p, k] * Data_State[p, k]
    return (Result_Matrix, Result_Value)
end

function Update_Cache!(Cache, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, true, 𝔽},
        X, Data...) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    Result_Matrix, Result_Value = Cache
    @tullio Result_Matrix[j, p, k] = X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] = Result_Matrix[j, p, k] * Data_State[p, k]
    return nothing
end

function Make_Cache(M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽},
        X, Data...) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show size(X), size(Phase)
    @tullio Result_Matrix[j, p, k] := X[p, j, q] * Phase[q, k]
    #     @show size(Result_Matrix), size(Data_State)
    @tullio Result_Value[j, k] := Result_Matrix[j, p, k] * Data_State[p, k]
    return (Result_Matrix, Result_Value)
end

function Update_Cache!(Cache, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽},
        X, Data...) where {Rows, Columns, Skew_Dimension, 𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    Result_Matrix, Result_Value = Cache
    @tullio Result_Matrix[j, p, k] = X[p, j, q] * Phase[q, k]
    @tullio Result_Value[j, k] = Result_Matrix[j, p, k] * Data_State[p, k]
    return nothing
end

function Evaluate!(Result, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, tall, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)) where {Rows, Columns, Skew_Dimension, tall, 𝔽}
    Result .= Cache[2]
    return nothing
end

function Evaluate_Add!(Result, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, tall, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)) where {Rows, Columns, Skew_Dimension, tall, 𝔽}
    Result .+= Cache[2]
    return nothing
end

function L0_DF!(DF, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, true, 𝔽}, X, Data::Vararg{AbstractMatrix{T}};
        L0, Cache = Make_Cache(M, X, Data...)) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[i, p, q] = L0[i, k] * Data_State[p, k] * Phase[q, k]
    return nothing
end

function L0_DF!(
        DF, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽}, X, Data::Vararg{AbstractMatrix{T}};
        L0, Cache = Make_Cache(M, X, Data...)) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[p, i, q] = L0[i, k] * Data_State[p, k] * Phase[q, k]
    return nothing
end

function Scaled_Hessian!(HH, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽}, X,
        Data::Vararg{AbstractMatrix{T}}; Scaling, Cache = nothing) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]
    Id = Diagonal(I, size(X, 2))
    @tullio HH[p1, i1, q1, p2, i2, q2] = Id[i1, i2] * Scaling[k] * Data_State[p1, k] * Phase[q1, k] *
                                         Data_State[p2, k] * Phase[q2, k]
end

# CALCULATE HH[p1, i1, q1, p2, i2, q2] = L0[i1, i2, k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k]
function L0_DF_DF_Delta!(DF, Delta, Latent_Delta, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽}, X,
        Data::Vararg{AbstractMatrix{T}}; Scaling, Cache = nothing) where {Rows, Columns, Skew_Dimension, 𝔽, T}
    Phase = Data[1]
    Data_State = Data[3]

    @tullio Latent_Delta[l, k] = Delta[p1, l, q1] * Data_State[p1, k] * Phase[q1, k]
    Latent_Delta .*= reshape(Scaling, 1, :)
    @tullio DF[p2, l, q2] = Latent_Delta[l, k] * Data_State[p2, k] * Phase[q2, k]
    #     println("L0_DF_DF_Delta!")
    #     @show norm(Delta), norm(Latent_Delta), norm(Data_State), norm(Phase)
    #     @show norm(DF)
    return nothing
end

function riemannian_Hessian!(M::ArrayStiefelOblique, Y, p, eG, eH, X)
    project!(M.manifold, Y, p, eH) #first term - project the Euclidean Hessian
    Y .+= Weingarten(M.manifold, p, X, eG - project(M.manifold, p, eG))
    return Y
end

function Jacobian_Add!(Jac, M::ArrayStiefelOblique{Rows, Columns, Skew_Dimension, false, 𝔽}, X, Data...;
        Cache = Make_Cache(M, X, Data...)) where {Rows, Columns, Skew_Dimension, 𝔽}
    Jac .+= Cache[1]
    nothing
end
