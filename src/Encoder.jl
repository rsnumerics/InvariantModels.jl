# SPDX-License-Identifier: EUPL-1.2

# (Linear, Nonlinear)
# Linear: Array_Stiefel, Mean_Stiefel, Stiefel_Oblique
# Nonlinear: Full, Projected, Local

"""
    @enum Encoder_Linear_Type begin
        Encoder_Fixed = 0
        Encoder_Array_Stiefel = 1
        Encoder_Mean_Stiefel = 2
        Encoder_Stiefel_Oblique = 3
        Encoder_Orthogonal = 4
    end

Enumeration to describe the linear part of the Encoder.
* `Encoder_Fixed` the encoder is not optimised, it is left fixed at its initial state
* `Encoder_Array_Stiefel` the encoder is a pointwise orthogonal set of matrices
* `Encoder_Mean_Stiefel` the encoder is orthogonal in an average sense over all possible forcing
* `Encoder_Stiefel_Oblique` the encoder consist of unit vectors for all points of forcing (not orthogonal)
* `Encoder_Orthogonal` the encoder is orthogonal to all linear parts of the other foliations and
    this orthogonality is periodically updated with the other foliations
"""
Encoder_Linear_Type
@enum Encoder_Linear_Type begin
    Encoder_Fixed = 0
    Encoder_Array_Stiefel = 1
    Encoder_Mean_Stiefel = 2
    Encoder_Stiefel_Oblique = 3
    Encoder_Orthogonal = 4
end

"""
    @enum Encoder_Nonlinear_Type begin
        Encoder_Dense_Full = 0
        Encoder_Dense_Latent_Linear = 1
        Encoder_Dense_Local = 2
        Encoder_Compressed_Full = 16
        Encoder_Compressed_Latent_Linear = 17
        Encoder_Compressed_Local = 18
    end

Enumeration to describe the nonlinear part of the Encoder
* `Encoder_Dense_Full` the encoder is a dense polynomial without any constraints
* `Encoder_Dense_Latent_Linear` the encoder is a dense polynomial, but it is linear for the latent variables
* `Encoder_Dense_Local` the encoder is a dense polynomial, but it is locally defined about an invariant manifold
* `Encoder_Compressed_Full` the encoder is a compressed polynomial without any constraints
* `Encoder_Compressed_Latent_Linear` the encoder is a compressed polynomial, but it is linear for the latent variables
* `Encoder_Compressed_Local` the encoder is a compressed polynomial, but it is locally defined about an invariant manifold
"""
Encoder_Nonlinear_Type
@enum Encoder_Nonlinear_Type begin
    Encoder_Dense_Full = 0
    Encoder_Dense_Latent_Linear = 1
    Encoder_Dense_Local = 2
    Encoder_Compressed_Full = 16
    Encoder_Compressed_Latent_Linear = 17
    Encoder_Compressed_Local = 18
end

struct QPEncoder{
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
} <: AbstractDecoratorManifold{𝔽}
    Orthogonal_Indices::Any
    manifold::ProductManifold{𝔽}
end

@inline Base.getindex(M::QPEncoder, i::Integer) = M.manifold[i]
@inline Base.length(M::QPEncoder) = length(M.manifold.manifolds)
ManifoldsBase.decorated_manifold(M::QPEncoder) = M.manifold
ManifoldsBase.active_traits(f, ::QPEncoder, args...) = ManifoldsBase.IsExplicitDecorator()

function Is_Linear_Fixed(Linear_Type::Encoder_Linear_Type)
    return ((Linear_Type == Encoder_Fixed) || (Linear_Type == Encoder_Orthogonal))
end

function Is_Linear_Fixed(
    MF::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    return Is_Linear_Fixed(Linear_Type)
end

function Is_Orthogonal(Linear_Type::Encoder_Linear_Type)
    return (Linear_Type == Encoder_Orthogonal)
end

function Is_Orthogonal(
    MF::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    return Is_Orthogonal(Linear_Type)
end

function Is_Model_Autonomous(Linear_Type::Encoder_Linear_Type)
    return (
        (Linear_Type == Encoder_Mean_Stiefel) || (Linear_Type == Encoder_Stiefel_Oblique)
    )
end

function Is_Model_Autonomous(
    MF::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    return Is_Model_Autonomous(Linear_Type)
end

function Is_Dense(Nonlinear_Type::Encoder_Nonlinear_Type)
    return (
        (Nonlinear_Type == Encoder_Dense_Full) ||
        (Nonlinear_Type == Encoder_Dense_Latent_Linear) ||
        (Nonlinear_Type == Encoder_Dense_Local)
    )
end

function QPEncoder(
    Latent_Dimension,
    State_Dimension,
    Orthogonal_Indices,
    Skew_Dimension,
    Encoder_Order,
    field::AbstractNumbers = ℝ;
    node_ratio = 1.0,
    leaf_ratio = 0.5,
    max_rank = 24,
    Linear_Type::Encoder_Linear_Type = Encoder_Array_Stiefel,
    Nonlinear_Type::Encoder_Nonlinear_Type = Encoder_Compressed_Latent_Linear,
)
    indices = unique(sort(Orthogonal_Indices))
    @assert issubset(indices, 1:State_Dimension) "Orthogonal_Indices does not belong to 1:State_Dimension ."

    if Nonlinear_Type == Encoder_Dense_Full
        Tensor_List = [
            Dense_Polynomial(
                Latent_Dimension,
                State_Dimension,
                Skew_Dimension,
                2,
                Encoder_Order,
            ),
        ]
    elseif Nonlinear_Type == Encoder_Dense_Latent_Linear
        Tensor_List = [
            Dense_Polynomial(
                Latent_Dimension,
                State_Dimension,
                Skew_Dimension,
                2,
                Encoder_Order;
                Perperdicular_Indices = Orthogonal_Indices,
            ),
        ]
    elseif Nonlinear_Type == Encoder_Compressed_Local
        # only view
        Tensor_List = [
            HTTensor(
                vcat(Skew_Dimension, repeat([length(Orthogonal_Indices)], k)),
                Latent_Dimension,
                node_ratio = node_ratio,
                leaf_ranks = max.(
                    round.(
                        Int,
                        vcat(
                            Skew_Dimension * leaf_ratio,
                            repeat([length(Orthogonal_Indices) * leaf_ratio], k),
                        ),
                    ),
                    1,
                ),
                max_rank = max_rank,
            ) for k = 2:Encoder_Order
        ]
    elseif Nonlinear_Type == Encoder_Compressed_Full
        # only full
        Tensor_List = [
            HTTensor(
                vcat(Skew_Dimension, repeat([State_Dimension], k)),
                Latent_Dimension,
                node_ratio = node_ratio,
                leaf_ranks = max.(
                    round.(
                        Int,
                        vcat(
                            Skew_Dimension * leaf_ratio,
                            repeat([State_Dimension * leaf_ratio], k),
                        ),
                    ),
                    1,
                ),
                max_rank = max_rank,
            ) for k = 2:Encoder_Order
        ]
    elseif Nonlinear_Type == Encoder_Compressed_Latent_Linear
        # view + (k-1) x full
        Tensor_List = [
            HTTensor(
                vcat(
                    Skew_Dimension,
                    length(Orthogonal_Indices),
                    repeat([State_Dimension], k - 1),
                ),
                Latent_Dimension,
                node_ratio = node_ratio,
                leaf_ranks = max.(
                    round.(
                        Int,
                        vcat(
                            Skew_Dimension * leaf_ratio,
                            length(Orthogonal_Indices) * leaf_ratio,
                            repeat([State_Dimension * leaf_ratio], k - 1),
                        ),
                    ),
                    1,
                ),
                max_rank = max_rank,
            ) for k = 2:Encoder_Order
        ]
    end
    if Linear_Type == Encoder_Array_Stiefel
        M = ProductManifold(
            SkewFunction(Latent_Dimension, Skew_Dimension),
            ArrayStiefel(Latent_Dimension, State_Dimension, Skew_Dimension, tall = false),
            Tensor_List...,
        )
    elseif Linear_Type == Encoder_Mean_Stiefel
        M = ProductManifold(
            SkewFunction(Latent_Dimension, Skew_Dimension),
            Mean_Flat_Stiefel(Latent_Dimension, State_Dimension, Skew_Dimension),
            Tensor_List...,
        )
    elseif Linear_Type == Encoder_Stiefel_Oblique
        M = ProductManifold(
            SkewFunction(Latent_Dimension, Skew_Dimension),
            ArrayStiefelOblique(
                Latent_Dimension,
                State_Dimension,
                Skew_Dimension,
                tall = false,
            ),
            Tensor_List...,
        )
    end
    return QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        field,
    }(
        indices,
        M,
    )
end

# this makes sure that the Linear part is the identity for the right indices
function Base.zero(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    X = ArrayPartition(map(zero, M.manifold.manifolds))
    Linear_Part = X.x[2]
    Linear_Part .= 0 # zero produces random for Stiefel
    Straight_Indices = setdiff(1:State_Dimension, M.Orthogonal_Indices)
    if Linear_Type == Encoder_Mean_Stiefel
        for j in axes(Linear_Part, 1), k in axes(Linear_Part, 3)
            Linear_Part[j, Straight_Indices[j], k] = 1
        end
    else
        for j in axes(Linear_Part, 2), k in axes(Linear_Part, 3)
            Linear_Part[Straight_Indices[j], j, k] = 1
        end
    end
    return X
end

function Slice(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Encoded_Phase,
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    M_List = []
    X_List = []
    for k in eachindex(X.x)
        MS, XS = Slice(M[k], X.x[k], Encoded_Phase)
        push!(M_List, MS)
        push!(X_List, XS)
    end
    manifold = ProductManifold(M_List...)
    X_Slice = ArrayPartition(X_List...)
    M_Slice = QPEncoder{
        Latent_Dimension,
        State_Dimension,
        1,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    }(
        M.Orthogonal_Indices,
        manifold,
    )
    return M_Slice, X_Slice
end

function Make_Cache(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data::Matrix,
    Encoded_Phase::Matrix,
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    # Eval for a HTtensor takes VarArg arguments, so we can write thata, Data
    Data_View = view(Data, M.Orthogonal_Indices, :)
    if Nonlinear_Type == Encoder_Compressed_Local
        return ArrayPartition(
            map(
                (m, x) -> Make_Cache(
                    m,
                    x,
                    Encoded_Phase,
                    Data_View,
                    ifelse(typeof(m) <: TensorManifold, Data_View, Data),
                ),
                M.manifold.manifolds,
                X.x,
            ),
        )
    elseif Nonlinear_Type == Encoder_Compressed_Full
        return ArrayPartition(
            map(
                (m, x) -> Make_Cache(m, x, Encoded_Phase, Data, Data),
                M.manifold.manifolds,
                X.x,
            ),
        )
    else
        return ArrayPartition(
            map(
                (m, x) -> Make_Cache(m, x, Encoded_Phase, Data_View, Data),
                M.manifold.manifolds,
                X.x,
            ),
        )
    end
end

function Update_Cache!(
    Cache,
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data::Matrix,
    Encoded_Phase::Matrix,
    sel,
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    Data_View = view(Data, M.Orthogonal_Indices, :)
    if length(sel) == 1
        #         println("QPEncoder update cache 1...")
        Update_Cache!(
            Cache.x[sel[1]],
            M[sel[1]],
            X.x[sel[1]],
            Encoded_Phase,
            Data_View,
            Data,
        )
    else
        #         println("QPEncoder update cache 2...")
        if Nonlinear_Type == Encoder_Compressed_Local
            Update_Cache_Parts!(
                Cache.x[sel[1]],
                M[sel[1]],
                X.x[sel[1]],
                Encoded_Phase,
                Data_View,
                ii = sel[2],
            )
        elseif Nonlinear_Type == Encoder_Compressed_Full
            Update_Cache_Parts!(
                Cache.x[sel[1]],
                M[sel[1]],
                X.x[sel[1]],
                Encoded_Phase,
                Data,
                Data,
                ii = sel[2],
            )
        else
            Update_Cache_Parts!(
                Cache.x[sel[1]],
                M[sel[1]],
                X.x[sel[1]],
                Encoded_Phase,
                Data_View,
                Data,
                ii = sel[2],
            )
        end
    end
    return nothing
end

function Update_Cache!(Cache, M::QPEncoder, X, Data::Matrix, Encoded_Phase::Matrix)
    Start_Component = Complete_Component_Index(X, (1,))
    Component = []
    push!(Component, Start_Component)
    while true
        Update_Cache!(Cache, M, X, Data::Matrix, Encoded_Phase::Matrix, Component[1])
        Component[1] = Next_Component(X, Component[1])
        if Component[1] == Start_Component
            break
        end
    end
    return nothing
end

function Evaluate!(
    Result,
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data::Matrix,
    Encoded_Phase::Matrix;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
    Lambda = 1,
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    Data_View = view(Data, M.Orthogonal_Indices, :)
    #     @time for (m, x, c) in zip(M.M.manifolds, X.x, Cache.x)
    #         @views Evaluate_Add!(Result, m, x, Encoded_Phase, Data_View, Data, Cache=c)
    #     end
    Result .= 0
    for k in eachindex(X.x)
        if k < 3
            Evaluate_Add!(
                Result,
                M[k],
                X.x[k],
                Encoded_Phase,
                Data_View,
                Data,
                Cache = Cache.x[k],
            )
        else
            if Nonlinear_Type == Encoder_Compressed_Local
                Evaluate_Add!(
                    Result,
                    M[k],
                    X.x[k],
                    Encoded_Phase,
                    Data_View,
                    Cache = Cache.x[k],
                    Lambda = Lambda,
                )
            elseif Nonlinear_Type == Encoder_Compressed_Full
                Evaluate_Add!(
                    Result,
                    M[k],
                    X.x[k],
                    Encoded_Phase,
                    Data,
                    Data,
                    Cache = Cache.x[k],
                    Lambda = Lambda,
                )
            else
                Evaluate_Add!(
                    Result,
                    M[k],
                    X.x[k],
                    Encoded_Phase,
                    Data_View,
                    Data,
                    Cache = Cache.x[k],
                    Lambda = Lambda,
                )
            end
        end
    end
    return nothing
end

function L0_DF_parts!(
    DF,
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase,
    Component;
    L0,
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    if length(Component) == 1
        #         println("L0_DF_parts! -> 1")
        return L0_DF!(
            DF,
            M[Component[1]],
            X.x[Component[1]],
            Encoded_Phase,
            view(Data, M.Orthogonal_Indices, :),
            Data,
            L0 = L0,
            Cache = Cache.x[Component[1]],
        )
    else
        if Nonlinear_Type == Encoder_Compressed_Local
            return L0_DF_parts!(
                DF,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                view(Data, M.Orthogonal_Indices, :),
                L0 = L0,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        elseif Nonlinear_Type == Encoder_Compressed_Full
            return L0_DF_parts!(
                DF,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                Data,
                Data,
                L0 = L0,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        else
            return L0_DF_parts!(
                DF,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                view(Data, M.Orthogonal_Indices, :),
                Data,
                L0 = L0,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        end
    end
end

function L0_DF_DF_Delta_parts!(
    DF,
    Delta,
    Latent_Delta,
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase,
    Component;
    Scaling,
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    if length(Component) == 1
        #         println("L0_DF_DF_Delta_parts! -> 1")
        return L0_DF_DF_Delta!(
            DF,
            Delta,
            Latent_Delta,
            M[Component[1]],
            X.x[Component[1]],
            Encoded_Phase,
            view(Data, M.Orthogonal_Indices, :),
            Data,
            Scaling = Scaling,
            Cache = Cache.x[Component[1]],
        )
    else
        if Nonlinear_Type == Encoder_Compressed_Local
            return L0_DF_DF_Delta_parts!(
                DF,
                Delta,
                Latent_Delta,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                view(Data, M.Orthogonal_Indices, :),
                Scaling = Scaling,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        elseif Nonlinear_Type == Encoder_Compressed_Full
            return L0_DF_DF_Delta_parts!(
                DF,
                Delta,
                Latent_Delta,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                Data,
                Data,
                Scaling = Scaling,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        else
            return L0_DF_DF_Delta_parts!(
                DF,
                Delta,
                Latent_Delta,
                M[Component[1]],
                X.x[Component[1]],
                Encoded_Phase,
                view(Data, M.Orthogonal_Indices, :),
                Data,
                Scaling = Scaling,
                ii = Component[2],
                Cache = Cache.x[Component[1]],
            )
        end
    end
end

function Jacobian!(
    Jac,
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
    Lambda = 1,
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    Data_View = view(Data, M.Orthogonal_Indices, :)
    Jac_View = view(Jac, :, M.Orthogonal_Indices, :)
    Jac .= 0
    for k in eachindex(X.x)
        if k < 3
            Jacobian_Add!(
                Jac,
                M[k],
                X.x[k],
                Encoded_Phase,
                Data_View,
                Data,
                Cache = Cache.x[k],
            )
        else
            if Is_Dense(Nonlinear_Type)
                Jacobian_Add!(
                    Jac,
                    M[k],
                    X.x[k],
                    Encoded_Phase,
                    Data_View,
                    Data,
                    Cache = Cache.x[k],
                    Lambda = Lambda,
                )
            elseif Nonlinear_Type == Encoder_Compressed_Local
                for d = 2:k
                    Jacobian_Add!(
                        Jac_View,
                        M[k],
                        X.x[k],
                        Encoded_Phase,
                        Data_View,
                        dim = d,
                        Cache = Cache.x[k],
                        Lambda = Lambda,
                    )
                end
            elseif Nonlinear_Type == Encoder_Compressed_Full
                for d = 2:k
                    Jacobian_Add!(
                        Jac,
                        M[k],
                        X.x[k],
                        Encoded_Phase,
                        Data,
                        dim = d,
                        Cache = Cache.x[k],
                        Lambda = Lambda,
                    )
                end
            else
                Jacobian_Add!(
                    Jac_View,
                    M[k],
                    X.x[k],
                    Encoded_Phase,
                    Data_View,
                    Data,
                    dim = 2,
                    Cache = Cache.x[k],
                    Lambda = Lambda,
                )
                for d = 3:k
                    Jacobian_Add!(
                        Jac,
                        M[k],
                        X.x[k],
                        Encoded_Phase,
                        Data_View,
                        Data,
                        dim = d,
                        Cache = Cache.x[k],
                        Lambda = Lambda,
                    )
                end
            end
        end
    end
    return nothing
end

function Jacobian_Test(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    Jac = zeros(Latent_Dimension, State_Dimension, size(Data, 2))
    Value = zeros(Latent_Dimension, size(Data, 2))
    Evaluate!(Value, M, X, Data, Encoded_Phase; Cache = Cache)
    Jacobian!(Jac, M, X, Data, Encoded_Phase; Cache = Cache)
    Data_FD = deepcopy(Data)
    Value_FD = deepcopy(Value)
    Jac_FD = deepcopy(Jac)
    Cache_FD = Make_Cache(M, X, Data_FD, Encoded_Phase)
    Eps = 1.0e-8
    for k = 1:State_Dimension
        Data_FD[k, :] .+= Eps
        Update_Cache!(Cache_FD, M, X, Data_FD, Encoded_Phase)
        Evaluate!(Value_FD, M, X, Data_FD, Encoded_Phase; Cache = Cache_FD)
        Jac_FD[:, k, :] .= (Value_FD - Value) ./ Eps
        Data_FD[k, :] .= Data[k, :]
        @show norm(Jac[:, k, :]), norm(Jac_FD[:, k, :] - Jac[:, k, :])
        @show Jac[:, k, 1:4]
        @show Jac_FD[:, k, 1:4]
    end
end

function Test_Loss!(
    Result,
    M::QPEncoder,
    X,
    Data::Matrix,
    Encoded_Phase::Matrix;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
)
    Evaluate!(Result, M, X, Data, Encoded_Phase; Cache = Cache)
    return real(sum(Result .* Result)) / 2
end

function Test_Gradient(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    L0 = zeros(Latent_Dimension, size(Data, 2))
    Result = zeros(Latent_Dimension, size(Data, 2))
    Result_FD = zeros(Latent_Dimension, size(Data, 2))
    X_FD = deepcopy(X)
    Cache_FD = Make_Cache(M, X_FD, Data, Encoded_Phase)
    Evaluate!(L0, M, X, Data, Encoded_Phase; Cache = Cache)
    #
    Start_Component = Complete_Component_Index(X, (1,))
    Component = []
    push!(Component, Start_Component)
    while true
        Point = Get_Component(X, Component[1])
        Point_FD = Get_Component(X_FD, Component[1])
        Grad = zero(Point)
        Grad_FD = zero(Point)
        L0_DF_parts!(Grad, M, X, Data, Encoded_Phase, Component[1]; L0 = L0, Cache = Cache)
        Loss = Test_Loss!(Result, M, X, Data, Encoded_Phase; Cache = Cache)
        Eps = 1.0e-8
        println(Component[1])
        Indices = randperm(length(Point_FD))[1:min(10, length(Point_FD))]
        for k in Indices #eachindex(Point_FD)
            Point_FD[k] += Eps
            Update_Cache!(Cache_FD, M, X_FD, Data, Encoded_Phase, Component[1])
            Loss_FD = Test_Loss!(Result_FD, M, X_FD, Data, Encoded_Phase; Cache = Cache_FD)
            Grad_FD[k] = (Loss_FD - Loss) / Eps
            Point_FD[k] = Point[k]
            #             print(@sprintf("| G=%.2e[s] : DG=%.2e ", Grad[k],  Grad[k] - Grad_FD[k]))
        end
        #         println()
        Update_Cache!(Cache_FD, M, X_FD, Data, Encoded_Phase, Component[1])
        println(
            Component[1],
            " -> G=",
            norm(Grad[Indices]),
            " diff=",
            norm(Grad[Indices] - Grad_FD[Indices]),
        )
        #
        Component[1] = Next_Component(X, Component[1])
        if Component[1] == Start_Component
            break
        end
    end
end

function Test_Hessian(
    M::QPEncoder{
        Latent_Dimension,
        State_Dimension,
        Skew_Dimension,
        Encoder_Order,
        Linear_Type,
        Nonlinear_Type,
        𝔽,
    },
    X,
    Data,
    Encoded_Phase;
    Cache = Make_Cache(M, X, Data, Encoded_Phase),
) where {
    Latent_Dimension,
    State_Dimension,
    Skew_Dimension,
    Encoder_Order,
    Linear_Type,
    Nonlinear_Type,
    𝔽,
}
    Scaling = rand(1, size(Data, 2))
    L0 = zeros(Latent_Dimension, size(Data, 2))
    L0_FD = zeros(Latent_Dimension, size(Data, 2))
    Latent_Delta = zeros(Latent_Dimension, size(Data, 2))
    Result = zeros(Latent_Dimension, size(Data, 2))
    Result_FD = zeros(Latent_Dimension, size(Data, 2))
    X_FD = deepcopy(X)
    Cache_FD = Make_Cache(M, X_FD, Data, Encoded_Phase)
    #
    Start_Component = Complete_Component_Index(X, (1,))
    Component = []
    push!(Component, Start_Component)
    while true
        Point = Get_Component(X, Component[1])
        Point_FD = Get_Component(X_FD, Component[1])
        Grad = zero(Point)
        Grad_FD = zero(Point)
        Hess = zero(Point)
        Hess_FD = zero(Point)
        Delta = randn(size(Point)...) / sqrt(prod(size(Point)))
        L0_DF_DF_Delta_parts!(
            Hess,
            Delta,
            Latent_Delta,
            M,
            X,
            Data,
            Encoded_Phase,
            Component[1];
            Scaling = Scaling,
            Cache = Cache,
        )
        Evaluate!(L0, M, X, Data, Encoded_Phase; Cache = Cache)
        L0_DF_parts!(
            Grad,
            M,
            X,
            Data,
            Encoded_Phase,
            Component[1];
            L0 = L0 .* Scaling,
            Cache = Cache,
        )
        Eps = 2^round(log2(1.0e-8))
        println(Component[1])
        Indices = randperm(length(Point_FD))[1:min(10, length(Point_FD))]
        for k in Indices # eachindex(Point_FD)
            Point_FD[k] += Eps
            Update_Cache!(Cache_FD, M, X_FD, Data, Encoded_Phase, Component[1])
            Evaluate!(L0_FD, M, X_FD, Data, Encoded_Phase; Cache = Cache_FD)
            L0_DF_parts!(
                Grad_FD,
                M,
                X_FD,
                Data,
                Encoded_Phase,
                Component[1];
                L0 = L0_FD .* Scaling,
                Cache = Cache_FD,
            )
            Hess_FD[k] = sum((Grad_FD - Grad) .* Delta) / Eps
            Point_FD[k] = Point[k]
            #             print(@sprintf("| H=%.2e[s] : DH=%.2e ", Hess[k],  Hess[k] - Hess_FD[k]))
        end
        #         println()
        Update_Cache!(Cache_FD, M, X_FD, Data, Encoded_Phase, Component[1])
        println(
            Component[1],
            " -> H=",
            norm(Hess[Indices]),
            " diff=",
            norm(Hess[Indices] - Hess_FD[Indices]),
        )
        #
        Component[1] = Next_Component(X, Component[1])
        if Component[1] == Start_Component
            break
        end
    end
end
