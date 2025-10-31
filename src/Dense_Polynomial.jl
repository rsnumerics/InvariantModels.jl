struct Dense_Polynomial{Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,End_Order,𝔽} <:
       AbstractDecoratorManifold{𝔽}
    manifold::Euclidean
    Monomial_Exponents::Array
    Admissible::Any
    Tensor_Admissible::Any
    Constant_Indices::Any
    Tensor_Constant_Indices::Any
    Linear_Indices::Any
    Tensor_Linear_Indices::Any
    D1row::Any
    D1col::Any
    D1val::Any
    D2row::Any
    D2col::Any
    D2val::Any
end

@inline Base.getindex(M::Dense_Polynomial, i::Integer) = M.manifold[i]
@inline Base.length(M::Dense_Polynomial) = length(M.manifold.manifolds)
@inline ManifoldsBase.decorated_manifold(M::Dense_Polynomial) = M.manifold

# Forward all functions by default to the decorated base manifold
@inline ManifoldsBase.get_forwarding_type(::Dense_Polynomial, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::Dense_Polynomial, ::Any, _) = ManifoldsBase.SimpleForwardingType()

function Dense_Polynomial(Output_Dimension, Input_Dimension, Skew_Dimension, Start_Order, End_Order;
    Perperdicular_Indices=1:Input_Dimension, field::AbstractNumbers=ℝ)
    # start with the constant
    Monomial_Exponents = Make_Monomial_Exponents(Input_Dimension, 0, End_Order)
    # filter out the not requested
    if length(unique(Perperdicular_Indices)) == Input_Dimension
        Admissible = sort(findall(vec(sum(Monomial_Exponents, dims=1)) .>= Start_Order))
    else
        println("Dense_Polynomial: Restricting monomials.")
        Admissible = findall(
            (dropdims(sum(Monomial_Exponents, dims=1), dims=1) .>= Start_Order) .&&
            (dropdims(sum(Monomial_Exponents[Perperdicular_Indices, :], dims=1), dims=1) .!= 0),
        )
    end
    #
    if !isempty(Admissible)
        if Admissible == range(first(Admissible), last(Admissible))
            Admissible = range(first(Admissible), last(Admissible))
        end
        # with Fourier
        All_Indices = reshape(collect(1:(size(Monomial_Exponents, 2)*Skew_Dimension)), :, size(Monomial_Exponents, 2))
        Tensor_Admissible = vec(All_Indices[:, Admissible])
        if Tensor_Admissible == range(first(Tensor_Admissible), last(Tensor_Admissible))
            Tensor_Admissible = range(first(Tensor_Admissible), last(Tensor_Admissible))
        end
        Indices = reshape(collect(1:(length(Admissible)*Skew_Dimension)), :, length(Admissible))
        Linear_Indices = findall(vec(sum(Monomial_Exponents[:, Admissible], dims=1)) .== 1)
    else
        Tensor_Admissible = []
        Indices = []
        Linear_Indices = []
    end
    if !isempty(Linear_Indices)
        if Linear_Indices == range(first(Linear_Indices), last(Linear_Indices))
            Linear_Indices = range(first(Linear_Indices), last(Linear_Indices))
        end
        Tensor_Linear_Indices = vec(Indices[:, Linear_Indices])
        if Tensor_Linear_Indices == range(first(Tensor_Linear_Indices), last(Tensor_Linear_Indices))
            Tensor_Linear_Indices = range(first(Tensor_Linear_Indices), last(Tensor_Linear_Indices))
        end
    else
        Tensor_Linear_Indices = []
    end
    Constant_Indices = findall(vec(sum(Monomial_Exponents[:, Admissible], dims=1)) .== 0)
    if !isempty(Constant_Indices)
        if Constant_Indices == range(first(Constant_Indices), last(Constant_Indices))
            Constant_Indices = range(first(Constant_Indices), last(Constant_Indices))
        end
        Tensor_Constant_Indices = vec(Indices[:, Constant_Indices])
        if Tensor_Constant_Indices == range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
            Tensor_Constant_Indices = range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
        end
    else
        Tensor_Constant_Indices = []
    end

    # @show Admissible, Tensor_Admissible, Tensor_Linear_Indices, Tensor_Constant_Indices
    # differentiation
    D1row, D1col, D1val = First_Derivative(Monomial_Exponents, Monomial_Exponents[:, Admissible])
    #     D2row, D2col, D2val = Second_Derivative(Monomial_Exponents, Monomial_Exponents[:, Admissible])
    D2row, D2col, D2val = 0, 0, 0

    # U, b, lambda
    Library_Dimension = length(Admissible)
    # 1. Model 2. Initial condition
    manifold = Euclidean(Input_Dimension, Skew_Dimension, Library_Dimension, field=field)
    return Dense_Polynomial{Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,End_Order,field}(
        manifold,
        Monomial_Exponents, Admissible, Tensor_Admissible,
        Constant_Indices, Tensor_Constant_Indices,
        Linear_Indices, Tensor_Linear_Indices,
        D1row, D1col, D1val, D2row, D2col, D2val)
end

function Base.zero(M::Dense_Polynomial{Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,
    End_Order,field}) where {Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,End_Order,field}
    Library_Dimension = length(M.Admissible)
    WW = zeros(Output_Dimension, Skew_Dimension, Library_Dimension)
    return WW
end

function Slice(
    M::Dense_Polynomial{Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,End_Order,field},
    X,
    Encoded_Phase,
) where {Output_Dimension,Input_Dimension,Skew_Dimension,Start_Order,End_Order,field}
    Library_Dimension = length(M.Admissible)
    manifold = Euclidean(Input_Dimension, 1, Library_Dimension, field=field)
    M_Slice = Dense_Polynomial{Output_Dimension,Input_Dimension,1,Start_Order,End_Order,field}(
        manifold,
        M.Monomial_Exponents,
        M.Admissible, M.Admissible,
        M.Constant_Indices, M.Constant_Indices,
        M.Linear_Indices, M.Linear_Indices,
        M.D1row, M.D1col, M.D1val, M.D2row, M.D2col, M.D2val,
    )
    X_Slice = zero(M_Slice)
    X_Slice_R = view(X_Slice, :, 1, :)
    @tullio X_Slice_R[i, k] = X[i, j, k] * Encoded_Phase[j]
    return M_Slice, X_Slice
end

# requires: Evaluate_Library() from multistepmodel-component.jl
function Make_Cache(M::Dense_Polynomial, X, Data...)
    Phase = Data[1]
    Data_State = Data[3]
    Monomial_Exponents = view(M.Monomial_Exponents, :, M.Admissible)
    Library_Dimension = size(Monomial_Exponents, 2)
    Monomials = Evaluate_Library(Monomial_Exponents, Data_State)
    @tullio Values[i, k] := X[i, p, q] * Phase[p, k] * Monomials[q, k]
    return (Values=Values, Monomials=Monomials)
end

# requires: Evaluate_Library!() from multistepmodel-component.jl
function Update_Cache!(Cache, M::Dense_Polynomial, X, Data...)
    Values = Cache.Values
    Monomials = Cache.Monomials
    Phase = Data[1]
    Data_State = Data[3]
    Monomial_Exponents = view(M.Monomial_Exponents, :, M.Admissible)
    Library_Dimension = size(Monomial_Exponents, 2)
    Evaluate_Library!(Monomials, Monomial_Exponents, Data_State)
    @tullio Values[i, k] = X[i, p, q] * Phase[p, k] * Monomials[q, k]
    return nothing
end

function Evaluate!(Result, M::Dense_Polynomial, X, Data...; Cache=Make_Cache(M, X, Data...), Lambda=1)
    Result .= Cache.Values
    return nothing
end

function Evaluate_Add!(Result, M::Dense_Polynomial, X, Data...; Cache=Make_Cache(M, X, Data...), Lambda=1)
    #     @show size(Result), size(Cache.Values)
    Result .+= Cache.Values
    return nothing
end

function L0_DF!(DF, M::Dense_Polynomial, X, Data...; L0, Cache=Make_Cache(M, X, Data...), Lambda=1)
    Phase = Data[1]
    Monomials = Cache.Monomials
    #     @show  size(DF), size(L0), size(Data_State), size(Phase)
    @tullio DF[i, p, q] = L0[i, k] * Monomials[q, k] * Phase[p, k]
    return nothing
end

# Calculate HH[i1, q1, i2, q2] = L0[i1, i2, k] * Phase[q1, k] * Phase[q2, k]
function L0_DF_DF_Delta!(
    DF, Delta, Latent_Delta, M::Dense_Polynomial, X, Data...; Scaling, Cache=Make_Cache(M, X, Data...))
    Phase = Data[1]
    Monomials = Cache.Monomials
    @tullio Latent_Delta[l, k] = Delta[l, p1, q1] * Monomials[q1, k] * Phase[p1, k]
    Latent_Delta .*= reshape(Scaling, 1, :)
    @tullio DF[l, p2, q2] = Latent_Delta[l, k] * Monomials[q2, k] * Phase[p2, k]
    return nothing
end

@inline riemannian_Hessian!(M::Dense_Polynomial, Y, p, G, H, X) = riemannian_Hessian!(M.manifold, Y, p, G, H, X)

# requires: Jacobian_Function_Helper!() from multistepmodel-component.jl
function Jacobian_Add!(Jac, M::Dense_Polynomial, X, Data...; Cache=nothing, Lambda=1)
    Phase = Data[1]
    Data_State = Data[3]
    All_Monomials = Evaluate_Library(M.Monomial_Exponents, Data_State)
    #     @show size(Jac), size(All_Monomials)
    Jacobian_Function_Helper!(Jac, X, All_Monomials, Phase, M.D1row, M.D1val, M.D1col)
    return nothing
end
