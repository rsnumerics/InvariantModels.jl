# SPDX-License-Identifier: EUPL-1.2

struct ArrayStiefel{Rows,Columns,Skew_Dimension,tall,𝔽} <: AbstractDecoratorManifold{𝔽}
    manifold::PowerManifold
end

function ArrayStiefel(
    Rows::Int,
    Columns::Int,
    Skew_Dimension::Int;
    field::AbstractNumbers=ℝ,
    tall=true,
)
    if tall
        manifold = PowerManifold(Stiefel(Rows, Columns, field), Skew_Dimension)
    else
        manifold = PowerManifold(Stiefel(Columns, Rows, field), Skew_Dimension)
    end
    return ArrayStiefel{Rows,Columns,Skew_Dimension,tall,field}(manifold)
end

@inline ManifoldsBase.decorated_manifold(M::ArrayStiefel) = M.manifold

# Forward all functions by default to the decorated base manifold
@inline ManifoldsBase.get_forwarding_type(::ArrayStiefel, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::ArrayStiefel, ::Any, _) = ManifoldsBase.SimpleForwardingType()

@inline ManifoldsBase.default_retraction_method(M::ArrayStiefel) =
    default_retraction_method(M.manifold)

function Base.zero(M)
    return rand(M.manifold)
end

function Slice(M::ArrayStiefel{Rows,Columns,Skew_Dimension,tall,𝔽}, X, Encoded_Slice) where {Rows,Columns,Skew_Dimension,tall,𝔽}
    M_Slice = ArrayStiefel(Rows, Columns, 1, field=𝔽, tall=tall)
    X_Slice = zero(M_Slice)
    X_Slice_View = view(X_Slice, :, :, 1)
    @tullio X_Slice_View[i, j] = X[i, j, k] * Encoded_Slice[k]
    return M_Slice, X_Slice
end

function Make_Cache(
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,true,𝔽},
    X,
    Data...,
) where {Rows,Columns,Skew_Dimension,𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show size(X), size(Phase)
    @tullio Result_Matrix[j, p, k] := X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] := Result_Matrix[j, p, k] * Data_State[p, k]
    return (Result_Matrix, Result_Value)
end

function Update_Cache!(
    Cache,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,true,𝔽},
    X,
    Data...,
) where {Rows,Columns,Skew_Dimension,𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    Result_Matrix, Result_Value = Cache
    @tullio Result_Matrix[j, p, k] = X[j, p, q] * Phase[q, k]
    @tullio Result_Value[j, k] = Result_Matrix[j, p, k] * Data_State[p, k]
    return nothing
end

function Make_Cache(
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data...,
) where {Rows,Columns,Skew_Dimension,𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show size(X), size(Phase)
    @tullio Result_Matrix[j, p, k] := X[p, j, q] * Phase[q, k]
    #     @show size(Result_Matrix), size(Data_State)
    @tullio Result_Value[j, k] := Result_Matrix[j, p, k] * Data_State[p, k]
    return (Result_Matrix, Result_Value)
end

function Update_Cache!(
    Cache,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data...,
) where {Rows,Columns,Skew_Dimension,𝔽}
    Phase = Data[1]
    Data_State = Data[3]
    Result_Matrix, Result_Value = Cache
    @tullio Result_Matrix[j, p, k] = X[p, j, q] * Phase[q, k]
    @tullio Result_Value[j, k] = Result_Matrix[j, p, k] * Data_State[p, k]
    return nothing
end

function Evaluate!(
    Result,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,tall,𝔽},
    X,
    Data...;
    Cache=Make_Cache(M, X, Data...),
) where {Rows,Columns,Skew_Dimension,tall,𝔽}
    Result .= Cache[2]
    return nothing
end

function Evaluate_Add!(
    Result,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,tall,𝔽},
    X,
    Data...;
    Cache=Make_Cache(M, X, Data...),
) where {Rows,Columns,Skew_Dimension,tall,𝔽}
    Result .+= Cache[2]
    return nothing
end

function L0_DF!(
    DF,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,true,𝔽},
    X,
    Data::Vararg{AbstractMatrix{T}};
    L0,
    Cache=Make_Cache(M, X, Data...),
) where {Rows,Columns,Skew_Dimension,𝔽,T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[i, p, q] = L0[i, k] * Data_State[p, k] * Phase[q, k]
    return nothing
end

function L0_DF!(
    DF,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data::Vararg{AbstractMatrix{T}};
    L0,
    Cache=Make_Cache(M, X, Data...),
) where {Rows,Columns,Skew_Dimension,𝔽,T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[p, i, q] = L0[i, k] * Data_State[p, k] * Phase[q, k]
    return nothing
end

function Scaled_Hessian!(
    HH,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data::Vararg{AbstractMatrix{T}};
    Scaling,
    Cache=nothing,
) where {Rows,Columns,Skew_Dimension,𝔽,T}
    Phase = Data[1]
    Data_State = Data[3]
    Id = Diagonal(I, size(X, 2))
    @tullio HH[p1, i1, q1, p2, i2, q2] =
        Id[i1, i2] *
        Scaling[k] *
        Data_State[p1, k] *
        Phase[q1, k] *
        Data_State[p2, k] *
        Phase[q2, k]
end

# CALCULATE HH[p1, i1, q1, p2, i2, q2] = L0[i1, i2, k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k]
function L0_DF_DF_Delta!(
    DF,
    Delta,
    Latent_Delta,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data::Vararg{AbstractMatrix{T}};
    Scaling,
    Cache=nothing,
) where {Rows,Columns,Skew_Dimension,𝔽,T}
    Phase = Data[1]
    Data_State = Data[3]
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    # CONSTANT -> DF[i2, q2] = L0[i1, i2, k] * theta[q1, k] * Delta[i1, q1] * theta[q2, k]
    # [p2, i2, q2, p1, i1, q1]
    #     println("ArrayStiefel : L0_DF_DF_Delta! - HH")
    #     @time @tullio HH[p1, q1, p2, q2] := Scaling[k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k]
    #     @time @tullio DF[p2, i2, q2] = HH[p1, q1, p2, q2] * Delta[p1, i2, q1]
    @tullio Latent_Delta[l, k] = Delta[p1, l, q1] * Data_State[p1, k] * Phase[q1, k]    #=@time=#
    Latent_Delta .*= reshape(Scaling, 1, :)
    @tullio DF[p2, l, q2] = Latent_Delta[l, k] * Data_State[p2, k] * Phase[q2, k]
    #     @tullio DF[p2, i2, q2] = L0[i1, i2, k] * Data_State[p1, k] * Phase[q1, k]* Data_State[p2, k] * Phase[q2, k] * Delta[p1, i1, q1]
    return nothing
end

function riemannian_Hessian!(M::ArrayStiefel, Y, p, eG, eH, X)
    project!(M.manifold, Y, p, eH) #first term - project the Euclidean Hessian
    Y .+= Weingarten(M.manifold, p, X, eG - project(M.manifold, p, eG))
    return Y
end

function Jacobian_Add!(
    Jac,
    M::ArrayStiefel{Rows,Columns,Skew_Dimension,false,𝔽},
    X,
    Data...;
    Cache=Make_Cache(M, X, Data...),
) where {Rows,Columns,Skew_Dimension,𝔽}
    Jac .+= Cache[1]
    nothing
end
