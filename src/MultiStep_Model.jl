function Make_Monomial_Exponents(State_Dimension, Start_Order, End_Order)
    Monomials_Per_Order =
        (length(multiexponents(State_Dimension, k)) for k = Start_Order:End_Order)
    Monomial_Index_List = (0, cumsum(Monomials_Per_Order)...)
    Monomial_Exponents = zeros(Int, State_Dimension, Monomial_Index_List[end])
    @inbounds for (j, order) in enumerate(Start_Order:End_Order)
        for (k, m) in enumerate(multiexponents(State_Dimension, order))
            Monomial_Exponents[:, Monomial_Index_List[j]+k] .= m
        end
    end
    return Monomial_Exponents
end

# returns the first index of mexp, which equals to iexp
# defined in polymethods.jl
function Find_Monomial_Index(mexp, iexp)
    return findfirst(dropdims(prod(mexp .== iexp, dims = 1), dims = 1))
end

function Evaluate_Library!(Monomials, Monomial_Exponents, Data)
    @inbounds for j in axes(Monomial_Exponents, 2), k in axes(Data, 2)
        Monomials[j, k] = 1.0
        for r in axes(Data, 1)
            Monomials[j, k] *= Data[r, k]^Monomial_Exponents[r, j]
        end
    end
    return Monomials
end

function Evaluate_Library(Monomial_Exponents, Data)
    Library_Dimension = size(Monomial_Exponents, 2)
    Monomials = zeros(eltype(Data), Library_Dimension, size(Data, 2))
    Evaluate_Library!(Monomials, Monomial_Exponents, Data)
    return Monomials
end

# creates matrices indexed by the variable differentiated against
#   row[k,r], col[k,r], val[k,r].
# where r is the variable differentiated against
# The input are
#   input[k] : k is the monomial ;; in the space of to_exp
# The output is
#   output[k, v] : k is the monomial, v is the variable ;; in the space of from_exp
# The result is
#   output[row[k,r], r] = val[k,r] * input[col[k,r]]
# The way to differentiate (1/2) x[k] * x[k], where k is the monomial
#   z[r] += x[row[k,r]] * val[k,r] * x[col[k,r]]
# Part of the Hessian...
#   output[row[k1,r1], r1] += val[k1, r1] * input[col[k1, r1]]
#   output[row[k2,r2], r2] += val[k2, r2] * input[col[k2, r2]]
# How do we multiply these?
#   h[r1, r2] += I[row[k1,r1], row[k2,r2]] * val[k1, r1] * val[k2, r2] * input[col[k1, r1]] * input[col[k2, r2]]#
# create a list
#   [k1, k2, r1, r2] for which row[k1,r1] = row[k2,r2]
function First_Derivative(to_exp, from_exp)
    rowA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    colA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    valA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    for var = 1:size(from_exp, 1)
        row = Array{Int,1}(undef, 0)
        col = Array{Int,1}(undef, 0)
        val = Array{Int,1}(undef, 0)
        for k = 1:size(from_exp, 2)
            id = copy(from_exp[:, k]) # this is a copy not a view
            if id[var] > 0
                id[var] -= 1
                x = Find_Monomial_Index(to_exp, id)
                if x != nothing
                    push!(row, k)
                    push!(col, x)
                    push!(val, from_exp[var, k])
                end
            end
        end
        rowA[var] = row
        colA[var] = col
        valA[var] = val
    end
    mxl = maximum([length(vv) for vv in rowA])
    row = ones(Int, mxl, length(rowA))
    col = ones(Int, mxl, length(rowA))
    val = zeros(Int, mxl, length(rowA))
    for r = 1:size(row, 2)
        #         @show size(row[:, r]), size(rowA[r])
        cl = length(rowA[r])
        row[1:cl, r] .= rowA[r]
        col[1:cl, r] .= colA[r]
        val[1:cl, r] .= valA[r]
    end
    return row, col, val
end

function Second_Derivative(to_exp, from_exp)
    rowA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    colA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    valA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    for p = 1:size(from_exp, 1), q = 1:size(from_exp, 1)
        row = Array{Int,1}(undef, 0)
        col = Array{Int,1}(undef, 0)
        val = Array{Int,1}(undef, 0)
        for k = 1:size(from_exp, 2)
            id = copy(from_exp[:, k]) # this is a copy not a view
            if id[p] > 0
                id[p] -= 1
                if id[q] > 0
                    id[q] -= 1
                    x = Find_Monomial_Index(to_exp, id)
                    if x != nothing
                        push!(row, k)
                        push!(col, x)
                        if p == q
                            push!(val, from_exp[p, k] * (from_exp[p, k] - 1))
                        else
                            push!(val, from_exp[p, k] * from_exp[q, k])
                        end
                    end
                end
            end
        end
        rowA[p, q] = row
        colA[p, q] = col
        valA[p, q] = val
    end
    mxl = maximum([length(vv) for vv in rowA])
    row = ones(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    col = ones(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    val = zeros(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    for p = 1:size(from_exp, 1), q = 1:size(from_exp, 1)
        cl = length(rowA[p, q])
        row[1:cl, p, q] .= rowA[p, q]
        col[1:cl, p, q] .= colA[p, q]
        val[1:cl, p, q] .= valA[p, q]
    end
    return row, col, val
end

# differece is -> SH[j, k] * BB[:, :, j]
function Transfer_Operator_Right(BB, SH)
    TR = zeros(eltype(BB), size(SH, 1), size(BB, 1), size(SH, 2), size(BB, 3))
    for j in axes(SH, 1), k in axes(SH, 2)
        TR[j, :, k, :] .= SH[j, k] * BB[:, j, :]
    end
    return reshape(TR, size(SH, 1) * size(BB, 1), size(SH, 2) * size(BB, 3))
end

# difference is -> SH[j, k] * BB[:, :, k]
function Transfer_Operator_Left(BB, SH)
    TR = zeros(eltype(BB), size(SH, 1), size(BB, 1), size(SH, 1), size(BB, 1))
    for j in axes(SH, 1), k in axes(SH, 2)
        TR[j, :, k, :] .= SH[j, k] * BB[:, k, :]
    end
    return reshape(TR, size(SH, 1) * size(BB, 1), size(SH, 1) * size(BB, 1))
end

# function differentialMinusDiagonalOperator(grid, B, omega)
#     SH = differentialOperator(grid, omega)
#     TR = zeros(eltype(B), size(B, 1) * size(B, 3), size(B, 2) * size(B, 3))
#     nd = size(B, 1)
#     for j in axes(SH, 1), k in axes(SH, 2)
#         TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .=
#             I[j, k] * B[:, :, k] - SH[j, k] * Diagonal(I, nd)
#     end
#     return TR
# end

function Transfer_Generator(Jac, Omega_ODE)
    State_Dimension = size(Jac, 1)
    Skew_Dimension = size(Jac, 2)
    Grid = Fourier_Grid(Skew_Dimension)
    Id_Skew = Diagonal(ones(State_Dimension))
    DD = differentialOperator(Grid)
    Big_Jac = zeros(Skew_Dimension, State_Dimension, Skew_Dimension, State_Dimension)
    @show DD
    for p in axes(Big_Jac, 1), q in axes(Big_Jac, 3)
        Big_Jac[p, :, q, :] .= Jac[:, p, :] * I[p, q] - Omega_ODE * Id_Skew * DD[p, q]
    end
    Big_Jac_RS =
        reshape(Big_Jac, State_Dimension * Skew_Dimension, State_Dimension * Skew_Dimension)
    return Big_Jac_RS
end

# this is always a complex polynomial even if the actual input is real
# the eigenvalues and eigenvectors are all complex
struct MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories} <:
       AbstractDecoratorManifold{ℂ}
    manifold::Euclidean
    SH::Array
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

@inline Base.getindex(M::MultiStep_Model, i::Integer) = M.manifold[i]
@inline Base.length(M::MultiStep_Model) = length(M.manifold.manifolds)
ManifoldsBase.decorated_manifold(M::MultiStep_Model) = M.manifold
ManifoldsBase.active_traits(f, ::MultiStep_Model, args...) =
    ManifoldsBase.IsExplicitDecorator()
#
ManifoldsBase.manifold_dimension(M::MultiStep_Model) = manifold_dimension(M.manifold)

# add Trajectories here...
"""
    MultiStep_Model(State_Dimension, Skew_Dimension, Start_Order, End_Order, Trajectories)

Create a polynomial model representation.
* `State_Dimension` dimensionality of the state space
* `Skew_Dimension` dimensionality of the forcin space
* `Start_Order` the smallest order monomial to include
* `End_Order` the greatest order monomial to include
* `Trajectories` number of trajectories to be represented

The model includes an identified initial condition for each fitted trajectory, which is different from the first point of the trajectory.
"""
function MultiStep_Model(
    State_Dimension,
    Skew_Dimension,
    Start_Order,
    End_Order,
    Trajectories,
)
    # start with the constant
    Monomial_Exponents = Make_Monomial_Exponents(State_Dimension, 0, End_Order)
    # filter out the not requested
    Admissible = sort(findall(vec(sum(Monomial_Exponents, dims = 1)) .>= Start_Order))
    if !isempty(Admissible)
        if Admissible == range(first(Admissible), last(Admissible))
            Admissible = range(first(Admissible), last(Admissible))
        end
        # with Fourier
        All_Indices = reshape(
            collect(1:(size(Monomial_Exponents, 2)*Skew_Dimension)),
            :,
            size(Monomial_Exponents, 2),
        )
        Tensor_Admissible = vec(All_Indices[:, Admissible])
        if Tensor_Admissible == range(first(Tensor_Admissible), last(Tensor_Admissible))
            Tensor_Admissible = range(first(Tensor_Admissible), last(Tensor_Admissible))
        end
        Indices =
            reshape(collect(1:(length(Admissible)*Skew_Dimension)), :, length(Admissible))
        Linear_Indices =
            findall(vec(sum(Monomial_Exponents[:, Admissible], dims = 1)) .== 1)
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
        if Tensor_Linear_Indices ==
           range(first(Tensor_Linear_Indices), last(Tensor_Linear_Indices))
            Tensor_Linear_Indices =
                range(first(Tensor_Linear_Indices), last(Tensor_Linear_Indices))
        end
    else
        Tensor_Linear_Indices = []
    end
    Constant_Indices = findall(vec(sum(Monomial_Exponents[:, Admissible], dims = 1)) .== 0)
    if !isempty(Constant_Indices)
        if Constant_Indices == range(first(Constant_Indices), last(Constant_Indices))
            Constant_Indices = range(first(Constant_Indices), last(Constant_Indices))
        end
        Tensor_Constant_Indices = vec(Indices[:, Constant_Indices])
        if Tensor_Constant_Indices ==
           range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
            Tensor_Constant_Indices =
                range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
        end
    else
        Tensor_Constant_Indices = []
    end

    @show Admissible, Tensor_Admissible, Tensor_Linear_Indices, Tensor_Constant_Indices
    # differentiation
    D1row, D1col, D1val =
        First_Derivative(Monomial_Exponents, Monomial_Exponents[:, Admissible])
    #     D2row, D2col, D2val = Second_Derivative(Monomial_Exponents, Monomial_Exponents[:, Admissible])
    D2row, D2col, D2val = 0, 0, 0

    # U, b, lambda
    Library_Dimension = length(Admissible)
    # 1. Model 2. Initial condition
    manifold = Euclidean(
        State_Dimension * Skew_Dimension * Library_Dimension +
        State_Dimension * Trajectories,
    )
    return MultiStep_Model{
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        Trajectories,
    }(
        manifold,
        zeros(Skew_Dimension, Skew_Dimension),
        Monomial_Exponents,
        Admissible,
        Tensor_Admissible,
        Constant_Indices,
        Tensor_Constant_Indices,
        Linear_Indices,
        Tensor_Linear_Indices,
        D1row,
        D1col,
        D1val,
        D2row,
        D2col,
        D2val,
    )
end

function Make_Similar(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    New_Trajectories::Integer,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    M_Copy = MultiStep_Model(
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        New_Trajectories,
    )
    X_Copy = zero(M_Copy)
    X_Copy.WW .= X.WW
    return M_Copy, X_Copy
end

function To_Non_Autonomous(
    M::MultiStep_Model{State_Dimension,1,Start_Order,End_Order,Trajectories},
    X,
    SH,
    Skew_Dimension::Integer,
) where {State_Dimension,Start_Order,End_Order,Trajectories}
    M_Copy = MultiStep_Model(
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        Trajectories,
    )
    M_Copy.SH .= SH
    X_Copy = zero(M_Copy)
    X_Copy.WW .= X.WW
    X_Copy.IC .= X.IC
    return M_Copy, X_Copy
end

function Slice(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Encoded_Slice,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    M_Slice = MultiStep_Model(State_Dimension, 1, Start_Order, End_Order, Trajectories)
    M_Slice.SH .= 1
    X_Slice = zero(M_Slice)
    @tullio WW[i, k] := X.WW[i, j, k] * Encoded_Slice[j]
    X_Slice.WW .= reshape(WW, size(X_Slice.WW)...)
    return M_Slice, X_Slice
end

function Base.zero(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    #         println("MultiStep_Model: zero")
    Library_Dimension = length(M.Admissible)
    WW = zeros(State_Dimension, Skew_Dimension, Library_Dimension)
    IC = zeros(State_Dimension, Trajectories)
    return ComponentVector(WW = WW, IC = IC)
end

function Least_Squares_DMD(Index_List, Data, Encoded_Phase, Scaling; Monomial_Exponents)
    State_Dimension = size(Data, 1)
    #     Monomial_Exponents = view(M.Monomial_Exponents,:, M.Admissible)
    Library_Dimension = size(Monomial_Exponents, 2)
    Monomials = zeros(eltype(Data), Library_Dimension, size(Data, 2))
    @time Evaluate_Library!(Monomials, Monomial_Exponents, Data)
    XX = zeros(
        eltype(Data),
        size(Encoded_Phase, 1),
        Library_Dimension,
        size(Encoded_Phase, 1),
        Library_Dimension,
    )
    YX = zeros(eltype(Data), Library_Dimension, size(Encoded_Phase, 1), Library_Dimension)
    SH_XX = zeros(eltype(Data), size(Encoded_Phase, 1), size(Encoded_Phase, 1))
    SH_YX = zeros(eltype(Data), size(Encoded_Phase, 1), size(Encoded_Phase, 1))

    for idx = 1:(length(Index_List)-1)
        Monomials_View_X = Monomials[:, (Index_List[idx]+1):(Index_List[idx+1]-1)]
        Phase_View_X = Encoded_Phase[:, (Index_List[idx]+1):(Index_List[idx+1]-1)]
        Monomials_View_Y = Monomials[:, (Index_List[idx]+2):Index_List[idx+1]]
        Phase_View_Y = Encoded_Phase[:, (Index_List[idx]+2):Index_List[idx+1]]
        Scaling_View = Scaling[(Index_List[idx]+1):(Index_List[idx+1]-1)]
        @tullio XX[p1, q1, p2, q2] +=
            Scaling_View[k] *
            Phase_View_X[p1, k] *
            Monomials_View_X[q1, k] *
            Phase_View_X[p2, k] *
            Monomials_View_X[q2, k]
        @tullio YX[q1, p2, q2] +=
            Scaling_View[k] *
            Monomials_View_Y[q1, k] *
            Phase_View_X[p2, k] *
            Monomials_View_X[q2, k]
        @tullio SH_XX[p1, p2] += Phase_View_X[p1, k] * Phase_View_X[p2, k]
        @tullio SH_YX[p1, p2] += Phase_View_Y[p1, k] * Phase_View_X[p2, k]
    end
    XXr = reshape(
        XX,
        size(Encoded_Phase, 1) * Library_Dimension,
        size(Encoded_Phase, 1) * Library_Dimension,
    )
    YXr = reshape(YX, Library_Dimension, size(Encoded_Phase, 1) * Library_Dimension)
    DMD = YXr * pinv(XXr)
    SH = SH_YX * pinv(SH_XX)
    #     BB = permutedims(reshape(DMD, Library_Dimension, size(Encoded_Phase, 1), Library_Dimension), (1,3,2))
    BB = reshape(DMD, Library_Dimension, size(Encoded_Phase, 1), Library_Dimension)
    return BB, SH, Monomials
end

function Find_Shift(Index_List, Data, Encoded_Phase)
    SH_XX = zeros(eltype(Data), size(Encoded_Phase, 1), size(Encoded_Phase, 1))
    SH_YX = zeros(eltype(Data), size(Encoded_Phase, 1), size(Encoded_Phase, 1))
    for idx = 1:(length(Index_List)-1)
        Phase_View_X = view(Encoded_Phase, :, (Index_List[idx]+1):(Index_List[idx+1]-1))
        Phase_View_Y = view(Encoded_Phase, :, (Index_List[idx]+2):Index_List[idx+1])
        @tullio SH_XX[p1, p2] += Phase_View_X[p1, k] * Phase_View_X[p2, k]
        @tullio SH_YX[p1, p2] += Phase_View_Y[p1, k] * Phase_View_X[p2, k]
    end
    SH = SH_YX * pinv(SH_XX)
    return SH
end

function Unitary_Model(Index_List, Encoded_Phase)
    SH_XX = zeros(eltype(Encoded_Phase), size(Encoded_Phase, 1), size(Encoded_Phase, 1))
    SH_YX = zeros(eltype(Encoded_Phase), size(Encoded_Phase, 1), size(Encoded_Phase, 1))
    for idx = 1:(length(Index_List)-1)
        Phase_View_X = view(Encoded_Phase, :, (Index_List[idx]+1):(Index_List[idx+1]-1))
        Phase_View_Y = view(Encoded_Phase, :, (Index_List[idx]+2):Index_List[idx+1])
        @tullio SH_XX[p1, p2] += Phase_View_X[p1, k] * Phase_View_X[p2, k]
        @tullio SH_YX[p1, p2] += Phase_View_Y[p1, k] * Phase_View_X[p2, k]
    end
    SH = SH_YX * pinv(SH_XX)
    return SH
end

function Least_Squares_Model(Index_List, Data, Encoded_Phase, Scaling; Monomial_Exponents)
    State_Dimension = size(Data, 1)
    #     Monomial_Exponents = view(M.Monomial_Exponents,:, M.Admissible)
    Library_Dimension = size(Monomial_Exponents, 2)
    Monomials = zeros(eltype(Data), Library_Dimension, size(Data, 2))
    @time Evaluate_Library!(Monomials, Monomial_Exponents, Data)
    XX = zeros(
        eltype(Data),
        size(Encoded_Phase, 1),
        Library_Dimension,
        size(Encoded_Phase, 1),
        Library_Dimension,
    )
    YX = zeros(eltype(Data), State_Dimension, size(Encoded_Phase, 1), Library_Dimension)

    for idx = 1:(length(Index_List)-1)
        Monomials_View_X = view(Monomials, :, (Index_List[idx]+1):(Index_List[idx+1]-1))
        Phase_View_X = view(Encoded_Phase, :, (Index_List[idx]+1):(Index_List[idx+1]-1))
        Data_View_Y = view(Data, :, (Index_List[idx]+2):Index_List[idx+1])
        Scaling_View = view(Scaling, (Index_List[idx]+1):(Index_List[idx+1]-1))
        @tullio XX[p1, q1, p2, q2] +=
            Scaling_View[k] *
            Phase_View_X[p1, k] *
            Monomials_View_X[q1, k] *
            Phase_View_X[p2, k] *
            Monomials_View_X[q2, k]
        @tullio YX[q1, p2, q2] +=
            Scaling_View[k] *
            Data_View_Y[q1, k] *
            Phase_View_X[p2, k] *
            Monomials_View_X[q2, k]
    end
    XXr = reshape(
        XX,
        size(Encoded_Phase, 1) * Library_Dimension,
        size(Encoded_Phase, 1) * Library_Dimension,
    )
    YXr = reshape(YX, State_Dimension, size(Encoded_Phase, 1) * Library_Dimension)
    DMD = YXr * pinv(XXr)
    BB = reshape(DMD, State_Dimension, size(Encoded_Phase, 1), Library_Dimension)
    SH = Unitary_Model(Index_List, Encoded_Phase)
    return BB, SH, Monomials
end

"""
    From_Data!(M::MultiStep_Model, X, Index_List, Data, Encoded_Phase, Scaling; Linear=false)

Creates a polynomial model from data given by `Index_List, Data, Encoded_Phase`.
The model is created using linear least squares and therefore not optimised.
This is to provide an initial condition for optimisation later on.
* `M`, `X` the model representation
* `Index_List`, `Data` and `Encoded_Phase` input data
* `Scaling` a scaling factor to represent the importance of each data point
* `Linear` if `true` all nonlinear terms are zeroed
"""
function From_Data!(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Linear = false,
)
    #     println("MultiStep_Model: From_Data!")
    Monomial_Exponents = view(M.Monomial_Exponents, :, M.Admissible) #=Linear ? view(M.Monomial_Exponents,:, M.Linear_Indices) : =#
    BB, SH, Monomials = Least_Squares_Model(
        Index_List,
        Data,
        Encoded_Phase,
        Scaling;
        Monomial_Exponents = Monomial_Exponents,
    )
    #     @show size(X.WW), size(BB)
    if Linear
        X.WW .= 0
        @show size(BB), size(X.WW)
        X.WW[:, :, M.Linear_Indices] .= BB[:, :, M.Linear_Indices]
    else
        X.WW .= BB
    end
    for p = 1:(length(Index_List)-1)
        X.IC[:, p] = Data[:, 1+Index_List[p]]
    end
    M.SH .= SH
    return nothing
end

function Extract_Steady_State(BB, SH, Monomial_Exponents)
    Library_Dimension = size(Monomial_Exponents, 2)
    Skew_Dimension = size(BB, 2)
    #
    Indices = reshape(collect(1:(Library_Dimension*Skew_Dimension)), :, Library_Dimension)
    Constant_Indices = findall(vec(sum(Monomial_Exponents, dims = 1)) .== 0)
    if Constant_Indices == range(first(Constant_Indices), last(Constant_Indices))
        Constant_Indices = range(first(Constant_Indices), last(Constant_Indices))
    end
    Tensor_Constant_Indices = vec(Indices[:, Constant_Indices])
    if Tensor_Constant_Indices ==
       range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
        Tensor_Constant_Indices =
            range(first(Tensor_Constant_Indices), last(Tensor_Constant_Indices))
    end
    Linear_Indices = findall(vec(sum(Monomial_Exponents, dims = 1)) .== 1)
    if Linear_Indices == range(first(Linear_Indices), last(Linear_Indices))
        Linear_Indices = range(first(Linear_Indices), last(Linear_Indices))
    end

    DMD = Transfer_Operator_Right(BB, SH)
    Total_Dimension = size(DMD, 1)
    A0_Range = setdiff(1:Total_Dimension, Tensor_Constant_Indices)
    A0 = DMD[A0_Range, A0_Range]
    F = eigen(SH)
    v0 = DMD[A0_Range, Tensor_Constant_Indices]
    Steady_State_All = zeros(Complex{eltype(v0)}, size(v0)...)
    @show size(v0), size(F.vectors)
    for k in eachindex(F.values)
        Steady_State_All[:, k] .= (F.values[k] * I - A0) \ (v0 * F.vectors[:, k])
    end
    Steady_State_All *= inv(F.vectors)
    Steady_State = dropdims(
        sum(reshape(real.(Steady_State_All), Skew_Dimension, :, Skew_Dimension), dims = 1),
        dims = 1,
    )
    Steady_State_Linear = Steady_State[Linear_Indices.-last(Constant_Indices), :]
    @tullio Steady_State_Linear_SH[i, k] := Steady_State_Linear[i, l] * SH[k, l]
    return Steady_State_Linear_SH
end

struct MultiStep_Model_Cache{
    State_Dimension,
    Skew_Dimension,
    Start_Order,
    End_Order,
    Trajectories,
}
    Values::Any
    Monomials::Any
    Residual::Any # = Values - Data
    Jac::Any
    Forward_Jac::Any
    BB_Jac::Any
    Hessian::Any # HH
    Gradient::Any # GG
end

function Evaluate_Trajectory!(
    Values,
    Monomials,
    BB,
    Monomial_Exponents,
    Admissible,
    IC,
    Encoded_Phase,
)
    # initial condition
    #     @show ()
    Values[:, 1] .= IC
    #     @show size(Monomials), size(Values), size(Monomial_Exponents)
    @inbounds for j in axes(Monomial_Exponents, 2)
        Monomials[j, 1] = 1.0
        for r in axes(Monomial_Exponents, 1)
            Monomials[j, 1] *= Values[r, 1]^Monomial_Exponents[r, j]
        end
    end
    @views A_Monomials = Monomials[Admissible, :]
    #     @show size(BB), size(Encoded_Phase), size(A_Monomials)
    @inbounds for k = 2:size(Values, 2)
        Values[:, k] .= 0
        for p in axes(BB, 1), q in axes(BB, 2), r in axes(BB, 3)
            Values[p, k] += BB[p, q, r] * Encoded_Phase[q, k-1] * A_Monomials[r, k-1]
        end
        for j in axes(Monomial_Exponents, 2)
            Monomials[j, k] = 1.0
            for r in axes(Monomial_Exponents, 1)
                Monomials[j, k] *= Values[r, k]^Monomial_Exponents[r, j]
            end
        end
    end
    return nothing
end

# simulate the model from initial condition IC
function Evaluate_Trajectory(M::MultiStep_Model, X, IC, Encoded_Phase)
    Library_Dimension = size(M.Monomial_Exponents, 2)
    Values = zeros(eltype(X), length(IC), size(Encoded_Phase, 2))
    Monomials = zeros(eltype(X), Library_Dimension, size(Encoded_Phase, 2))
    Evaluate_Trajectory!(
        Values,
        Monomials,
        X.WW,
        M.Monomial_Exponents,
        M.Admissible,
        IC,
        Encoded_Phase,
    )
    return Values
end

function Evaluate!(
    Values,
    Monomials,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
)
    #     println("MultiStep_Model: Evaluate!")
    #     Library_Dimension = length(M.Admissible)
    #     Monomials = zeros(eltype(Values), Library_Dimension, size(Data, 2))
    @inbounds for t = 1:(length(Index_List)-1)
        @views Values_R = Values[:, (1+Index_List[t]):Index_List[t+1]]
        @views Monomials_R = Monomials[:, (1+Index_List[t]):Index_List[t+1]]
        @views Encoded_Phase_R = Encoded_Phase[:, (1+Index_List[t]):Index_List[t+1]]
        IC_R = X.IC[:, t]
        #         @show size(Monomials_R), size(Monomials)
        #         print("EVT ")
        Evaluate_Trajectory!(
            Values_R,
            Monomials_R,
            X.WW,
            M.Monomial_Exponents,
            M.Admissible,
            IC_R,
            Encoded_Phase_R,
        )        #=@time @views=#
        #         @show Values_R
    end
    #     return Monomials
    return nothing
end

"""
    Evaluate(M::MultiStep_Model, X, Index_List, Data, Encoded_Phase)

Applies the polynomial representation of the model `M`, `X` to the data `Index_List, Data, Encoded_Phase`
as if each data point was an initial condition.
"""
function Evaluate(M::MultiStep_Model, X, Index_List, Data, Encoded_Phase)
    #     println("MultiStep_Model: Evaluate")
    Library_Dimension = size(M.Monomial_Exponents, 2)
    Values = zeros(eltype(X), size(Data)...)
    Monomials = zeros(eltype(X), Library_Dimension, size(Data, 2))
    Evaluate!(Values, Monomials, M, X, Index_List, Data, Encoded_Phase)
    return Values
end

function Evaluate_Function!(Value, M::MultiStep_Model, X, Data, Encoded_Phase)
    Library_Dimension = length(M.Admissible)
    Monomials = zeros(eltype(Data), Library_Dimension, size(Data, 2))
    Evaluate_Library!(Monomials, M.Monomial_Exponents[:, M.Admissible], Data)
    X_Model = X.WW
    if size(X_Model, 2) == 1
        X_Model_V = view(X_Model, :, 1, :)
        @tullio Value[i, k] = X_Model_V[i, q] * Monomials[q, k]
    else
        @tullio Value[i, k] = X_Model[i, p, q] * Encoded_Phase[p, k] * Monomials[q, k]
    end
    return nothing
end

function MultiStep_Jacobian_Helper!(Jac, X, Monomials, Encoded_Phase, D1row, D1val, D1col)
    @inbounds @fastmath for r in axes(D1row, 2),
        p in axes(D1row, 1),
        q in axes(Encoded_Phase, 1),
        i in axes(X, 1)

        a = (X[i, q, D1row[p, r]] * D1val[p, r])
        for k in axes(Encoded_Phase, 2)
            Jac[i, r, k] += a * Monomials[D1col[p, r], k] * Encoded_Phase[q, k]
        end
    end

    nothing
end

# Specialise to linear systems
# function Evaluate_Jacobian_Step_One!(Cache::MultiStep_Model_Cache, M::MultiStep_Model{State_Dimension, Skew_Dimension, Start_Order, 1, Trajectories}, X,
#                                      Index_List, Data, Encoded_Phase, Scaling
#                                     ) where {State_Dimension, Skew_Dimension, Start_Order, Trajectories}
#     Jac = Cache.Jac
#     Forward_Jac = Cache.Forward_Jac
#     X_Model = view(X.WW, :, :, M.Linear_Indices)
#     @tullio Jac[i, j, k] = X_Model[i, p, j] * Encoded_Phase[p, k]
#     Forward_Jac .= 0 # = zero(Jac)
#     @inbounds for t in 1:length(Index_List) - 1
#         # Forward
#         for k in axes(Forward_Jac, 1)
#             Forward_Jac[k, 1+Index_List[t], k] = 1
#         end
#         for u in 2+Index_List[t]:Index_List[t+1]
#             Forward_Jac[:, u, :] .= (Jac[:, :, u] * Forward_Jac[:, u - 1, :])
#         end
#     end
#     return nothing
# end

# Full Monomials -> from constant term
function Evaluate_Jacobian_Step_One!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    #     println("MultiStep_Model: Evaluate_Jacobian_Step_One!")
    A_Monomials = Cache.Monomials # [M.Admissible, :]
    Jac = Cache.Jac
    Forward_Jac = Cache.Forward_Jac
    Jac .= 0
    Forward_Jac .= 0 # = zero(Jac)
    @inbounds for t = 1:(length(Index_List)-1)
        @views Jac_R = Jac[:, :, (2+Index_List[t]):Index_List[t+1]]
        @views A_Monomials_R = A_Monomials[:, (1+Index_List[t]):(Index_List[t+1]-1)]
        @views Encoded_Phase_R = Encoded_Phase[:, (1+Index_List[t]):(Index_List[t+1]-1)]
        for k in axes(Jac, 1)
            Jac[k, k, 1+Index_List[t]] = 1
        end
        MultiStep_Jacobian_Helper!(
            Jac_R,
            X.WW,
            A_Monomials_R,
            Encoded_Phase_R,
            M.D1row,
            M.D1val,
            M.D1col,
        )
        # Forward
        for k in axes(Forward_Jac, 1)
            Forward_Jac[k, 1+Index_List[t], k] = 1
        end
        for u = (2+Index_List[t]):Index_List[t+1]
            Forward_Jac[:, u, :] .= (Jac[:, :, u] * Forward_Jac[:, u-1, :])
        end
    end
    if any(isnan.(Data))
        println("Evaluate_Jacobian_Step_One!: Not a number in Data.")
    end
    if any(isnan.(Encoded_Phase))
        println("Evaluate_Jacobian_Step_One!: Not a number in Encoded_Phase.")
    end
    if any(isnan.(Cache.Values))
        println("Evaluate_Jacobian_Step_One!: Not a number in Values.")
        X.WW .= 0
        Cache.Values .= 0
    end
    if any(isnan.(A_Monomials))
        println("Evaluate_Jacobian_Step_One!: Not a number in Monomials.")
        X.WW .= 0
        A_Monomials .= 0
    end
    if any(isnan.(Jac))
        println("Evaluate_Jacobian_Step_One!: Not a number in Jacobian.")
        X.WW .= 0
        Jac .= 0
    end
    return nothing
end

# no need to be different for linear systems. The Monomials can be constructed differently, though
function Evaluate_Jacobian_Step_Two!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    #     println("MultiStep_Model: Evaluate_Jacobian_Step_Two!")
    A_Monomials = view(Cache.Monomials, M.Admissible, :)
    Jac = Cache.Jac
    Forward_Jac = Cache.Forward_Jac
    BB_Jac = Cache.BB_Jac
    Id = Diagonal(ones(size(Data, 1)))
    @inbounds for t = 1:(length(Index_List)-1)
        for u = (2+Index_List[t]):Index_List[t+1]
            Jac_R = view(Jac, :, :, u)
            BB_Jac_R = view(BB_Jac, :, u - 1, :, :, :)
            BB_Jac_C = view(BB_Jac, :, u, :, :, :)
            Encoded_Phase_R = view(Encoded_Phase, :, u - 1)
            A_Monomials_R = view(A_Monomials, :, u - 1)
            #             @show size(BB_Jac_C), size(A_Monomials_R), size(A_Monomials)
            @tullio BB_Jac_C[k1, l1, l2, l3] = Jac_R[k1, p] * BB_Jac_R[p, l1, l2, l3]
            @tullio BB_Jac_C[k1, l1, l2, l3] +=
                Id[k1, l1] * Encoded_Phase_R[l2] * A_Monomials_R[l3]
        end
    end
    AA_RS = reshape(BB_Jac, size(Data)..., :)
    #     BB_RS = deepcopy(Forward_Jac)
    #     @show size(AA_RS), size(BB_RS)
    return AA_RS, Forward_Jac
end

function Gradient_Helper(GG1, BB_Jac, Jac, A_Monomials, Encoded_Phase, Residual, Scaling, u)
    #     Jac_R = view(Jac, :, :, u)
    #     BB_Jac_R = view(BB_Jac, :, :, :, :, 1+mod(u-1, 2))
    #     BB_Jac_C = view(BB_Jac, :, :, :, :, 1+mod(u, 2))
    #     Encoded_Phase_R = view(Encoded_Phase, :, u-1)
    #     A_Monomials_R = view(A_Monomials, :, u-1)
    #     @tullio BB_Jac_C[k1, l1, l2, l3] = Jac_R[k1, p] * BB_Jac_R[p, l1, l2, l3]
    #     @tullio BB_Jac_C[k1, l1, l2, l3] += Id[k1, l1] * Encoded_Phase_R[l2] * A_Monomials_R[l3]
    #     Sc = Scaling[u]
    #     Residual_C = view(Residual, :, u)
    #     @tullio GG1[l1, l2, l3] += BB_Jac_C[p, l1, l2, l3] * Residual_C[p] * Sc
    idR = 1 + mod(u - 1, 2)
    idC = 1 + mod(u, 2)
    @inbounds @fastmath for k1 in axes(BB_Jac, 1),
        l1 in axes(BB_Jac, 2),
        l2 in axes(BB_Jac, 3),
        l3 in axes(BB_Jac, 4),
        p in axes(Jac, 1)

        BB_Jac[k1, l1, l2, l3, idC] = Jac[k1, p, idR] * BB_Jac[p, l1, l2, l3, idR]
    end
    @inbounds @fastmath for k1 in axes(BB_Jac, 1),
        l2 in axes(BB_Jac, 3),
        l3 in axes(BB_Jac, 4)

        BB_Jac[k1, k1, l2, l3, idC] += Encoded_Phase[l2, u-1] * A_Monomials[l3, u-1]
    end
    Sc = Scaling[u]
    @inbounds @fastmath for p in axes(BB_Jac, 1),
        l1 in axes(BB_Jac, 2),
        l2 in axes(BB_Jac, 3),
        l3 in axes(BB_Jac, 4)

        GG1[l1, l2, l3] += BB_Jac[p, l1, l2, l3, idC] * Residual[p, u] * Sc
    end
    nothing
end

# no need to be different for linear systems. The Monomials can be constructed differently, though
function Gradient!(
    GG,
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    #     println("MultiStep_Model: Evaluate_Jacobian_Step_Two!")
    # Update Cache
    Evaluate!(Cache.Values, Cache.Monomials, M, X, Index_List, Data, Encoded_Phase)
    Cache.Residual .= Cache.Values .- Data
    # start calculating
    Evaluate_Jacobian_Step_One!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    Residual = Cache.Residual
    A_Monomials = view(Cache.Monomials, M.Admissible, :)
    Jac = Cache.Jac
    Forward_Jac = Cache.Forward_Jac
    #     GG = Cache.Gradient
    GG .= 0
    S1 = prod(size(X.WW))
    S2 = size(Forward_Jac, 3)
    GG1 = reshape(view(GG, 1:S1), size(X.WW)...)
    Id = Diagonal(ones(size(Data, 1)))
    BB_Jac = zeros(eltype(Data), State_Dimension, size(X.WW)..., 2)
    @inbounds for t = 1:(length(Index_List)-1)
        for u = (2+Index_List[t]):Index_List[t+1]
            Gradient_Helper(
                GG1,
                BB_Jac,
                Jac,
                A_Monomials,
                Encoded_Phase,
                Residual,
                Scaling,
                u,
            )
        end
        R_range = (1+Index_List[t]):Index_List[t+1]
        C_range = (S1+1+(t-1)*S2):(S1+t*S2)
        Scaling_R = view(Scaling, R_range)
        BB_RS_Rt = view(Forward_Jac, :, R_range, :)
        #
        GG2 = view(GG, C_range)
        Residual_R = view(Residual, :, R_range)
        @tullio GG2[j] = Residual_R[p, k] * BB_RS_Rt[p, k, j] * Scaling_R[k]
    end
    return GG
end

function Gradient_Hessian!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
)
    #     println("MultiStep_Model: Gradient_Hessian!")
    Evaluate_Jacobian_Step_One!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    AA_RS, BB_RS =
        Evaluate_Jacobian_Step_Two!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    #
    Residual = Cache.Residual
    S1 = size(AA_RS, 3)
    S2 = size(BB_RS, 3)
    NT = length(Index_List) - 1
    HH = Cache.Hessian # zeros(S1 + NT * S2, S1 + NT * S2)
    GG = Cache.Gradient # zeros(S1 + NT * S2)
    begin
        HH11 = view(HH, 1:S1, 1:S1)
        #         mul!(HH11, transpose(AA_RS), AA_RS, 1, 0)
        @tullio HH11[i, j] = AA_RS[p, k, i] * AA_RS[p, k, j] * Scaling[k]
        GG1 = view(GG, 1:S1)
        #         mul!(GG1, transpose(AA_RS), vec(Residual), 1, 0)
        @tullio GG1[j] = Residual[p, k] * AA_RS[p, k, j] * Scaling[k]
    end
    for t = 1:NT
        R_range = (1+Index_List[t]):Index_List[t+1]
        C_range = (S1+1+(t-1)*S2):(S1+t*S2)
        Scaling_R = view(Scaling, R_range)
        AA_RS_Rt = view(AA_RS, :, R_range, :)
        BB_RS_Rt = view(BB_RS, :, R_range, :)
        HH12 = view(HH, C_range, 1:S1)
        #         mul!(HH12, transpose(BB_RS_Rt), AA_RS_Rt, 1, 0)
        @tullio HH12[i, j] = BB_RS_Rt[p, k, i] * AA_RS_Rt[p, k, j] * Scaling_R[k]
        #         HH12 .= BB_RS_Rt' * AA_RS_Rt
        HH[1:S1, C_range] .= transpose(HH12)
        HH22 = view(HH, C_range, C_range)
        #         mul!(HH22, transpose(BB_RS_Rt), BB_RS_Rt, 1, 0)
        @tullio HH22[i, j] = BB_RS_Rt[p, k, i] * BB_RS_Rt[p, k, j] * Scaling_R[k]
        #         HH22 .= BB_RS_Rt' * BB_RS_Rt
        #
        GG2 = view(GG, C_range)
        Residual_R = view(Residual, :, R_range)
        @tullio GG2[j] = Residual_R[p, k] * BB_RS_Rt[p, k, j] * Scaling_R[k]
    end
    #     display(AA_RS)
    return GG, HH
end

function Optimise!(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
    check = false,
    Radius = 0,
    Maximum_Radius = 4 * sqrt(manifold_dimension(M)),
)
    Beta = Radius
    LL = Loss(M, X, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
    GG, HH = Gradient_Hessian!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    XC = deepcopy(X)
    while true
        #         if check
        #             JJ = ForwardDiff.jacobian(x -> Evaluate(M, x, Index_List, Data, Encoded_Phase), X)
        #             JJS = reshape(JJ, size(Data)..., :)
        #             @tullio HH2[i, j] := JJS[p, k, i] * JJS[p, k, j] * Scaling[k]
        #             @show norm(HH - HH2)
        #             display(HH)
        #             display(JJ)
        #             return nothing
        #         end
        #         @show Beta
        DD = Diagonal(0.1 .+ diag(HH))
        XC .= X .- (HH + Beta * DD) \ GG
        Evaluate!(Cache.Values, Cache.Monomials, M, XC, Index_List, Data, Encoded_Phase)
        Cache.Residual .= Cache.Values .- Data
        LL2 = Loss(M, XC, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
        #         @show LL2
        #         println("Beta = ", Beta, " Loss = ", LL2, " Starting Loss = ", LL)
        if LL2 < 0.875 * LL
            X .= XC
            return Beta / 2, LL2, norm(GG)
        elseif LL2 <= LL + eps(LL)
            X .= XC
            return Beta / sqrt(2.0), LL2, norm(GG)
        elseif LL2 < 1.125 * LL
            #             X .= XC
            Beta = Beta * sqrt(2.0)
            #             return Beta * sqrt(2.0), LL2, norm(GG)
        elseif Beta < Maximum_Radius
            Beta = Beta * 3
        else
            return Beta, LL, norm(GG)
        end
    end
    #
    X .= XC
    return Beta, LL, norm(GG)
end

function Gradient_Hessian_IC!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
)
    Evaluate_Jacobian_Step_One!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    BB_RS = Cache.Forward_Jac
    if any(isnan.(BB_RS))
        println("Gradient_Hessian_IC!: Not a number in Jacobian.")
    end
    Residual = Cache.Residual
    NT = length(Index_List) - 1
    HH22 = zeros(eltype(Data), size(Data, 1), size(Data, 1), NT)
    GG2 = zeros(eltype(Data), size(Data, 1), NT)
    for t = 1:NT
        R_range = (1+Index_List[t]):Index_List[t+1]
        Scaling_R = view(Scaling, R_range)
        BB_RS_Rt = view(BB_RS, :, R_range, :)
        # Hessian
        HH22_R = view(HH22, :, :, t)
        @tullio HH22_R[i, j] = BB_RS_Rt[p, k, i] * BB_RS_Rt[p, k, j] * Scaling_R[k]
        # Gradient
        Residual_R = view(Residual, :, R_range)
        GG2_R = view(GG2, :, t)
        @tullio GG2_R[j] = Residual_R[p, k] * BB_RS_Rt[p, k, j] * Scaling_R[k]
    end
    return GG2, HH22
end

function Optimise_IC!(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
    check = false,
    Radius = 0,
    Maximum_Radius = 4 * sqrt(manifold_dimension(M)),
)
    #
    Beta = Radius
    LL = Loss(M, X, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
    GG, HH = Gradient_Hessian_IC!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
    if any(isnan.(GG)) || any(isnan.(HH))
        return Beta, LL, norm(GG)
    end
    XC = deepcopy(X)
    while true
        #         if check
        #             JJ = ForwardDiff.jacobian(x -> Evaluate(M, x, Index_List, Data, Encoded_Phase), X)
        #             JJS = reshape(JJ, size(Data)..., :)
        #             @tullio HH2[i, j] := JJS[p, k, i] * JJS[p, k, j] * Scaling[k]
        #             @show norm(HH - HH2)
        #             display(HH)
        #             display(JJ)
        #             return nothing
        #         end
        #         @show Beta
        for k in axes(HH, 3)
            DD = Diagonal(0.1 .+ diag(HH[:, :, k]))
            Delta = (HH[:, :, k] + Beta * DD) \ GG[:, k]
            if any(isnan.(Delta))
                return Beta, LL, norm(GG)
            end
            XC.IC[:, k] .= X.IC[:, k] .- Delta
        end
        Evaluate!(Cache.Values, Cache.Monomials, M, XC, Index_List, Data, Encoded_Phase)
        Cache.Residual .= Cache.Values .- Data
        LL2 = Loss(M, XC, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
        #         @show LL2
        #         println("Beta = ", Beta, " Loss = ", LL2, " Starting Loss = ", LL)
        if LL2 < 0.875 * LL
            X.IC .= XC.IC
            return Beta / 2, LL2, norm(GG)
        elseif LL2 <= LL + eps(LL)
            X.IC .= XC.IC
            return Beta / sqrt(2.0), LL2, norm(GG)
        elseif LL2 < 1.125 * LL
            #             X.IC .= XC.IC
            Beta = Beta * sqrt(2.0)
            #             return Beta * sqrt(2.0), LL2, norm(GG)
        elseif Beta < Maximum_Radius
            Beta = Beta * 3
        else
            return Beta, LL, norm(GG)
        end
    end
    #
    X.IC .= XC.IC
    return Beta, LL, norm(GG)
end

function Optimise_IC_Full!(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
    check = false,
    Radius = 4 * sqrt(manifold_dimension(M)) / 256,
    Maximum_Radius = 4 * sqrt(manifold_dimension(M)),
    Iterations = 32,
)
    t0 = time()
    Trust_Radius = deepcopy(Radius)
    if any(isnan.(Data))
        @show findall(isnan.(Data))
        return nothing
    end
    if any(isnan.(X))
        @show findall(isnan.(X))
        return nothing
    end
    for it = 1:Iterations
        Trust_Radius, M_Loss, M_Grad = Optimise_IC!(
            M,
            X,
            Index_List,
            Data,
            Encoded_Phase,
            Scaling,
            Cache = Cache,
            Radius = Trust_Radius,
            Maximum_Radius = Maximum_Radius,
        )
        txt = (
            "    Model IC -> $(it). " *
            @sprintf("time = %.1f[s] ", time() - t0) *
            @sprintf("F(x) = %.5e ", M_Loss) *
            @sprintf("G(x) = %.5e ", M_Grad) *
            @sprintf("R = %.5e ", Trust_Radius)
        )
        println(txt)
        if Trust_Radius >= Maximum_Radius
            Trust_Radius = 1.0
            return Trust_Radius
        end
    end
    return Trust_Radius
end

# function Gradient_IC!(GG, M::MultiStep_Model, X, Index_List, Data, Encoded_Phase, Scaling; Cache::MultiStep_Model_Cache)
#     Evaluate_Jacobian_Step_One!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
#     BB_RS = Cache.Forward_Jac
#     Residual = Cache.Residual
#     NT = length(Index_List) - 1
#     GG2 = reshape(GG, size(Data, 1), NT)
#     for t in 1:NT
#         R_range = 1 + Index_List[t] : Index_List[t+1]
#         Scaling_R = view(Scaling, R_range)
#         BB_RS_Rt = view(BB_RS, :, R_range, :)
#         # Gradient
#         Residual_R = view(Residual, :, R_range)
#         GG2_R = view(GG2, :, t)
#         @tullio GG2_R[j] = Residual_R[p, k] * BB_RS_Rt[p, k, j] * Scaling_R[k]
#     end
#     return GG
# end
#
# function Hessian_IC!(HH, M::MultiStep_Model, X, Index_List, Data, Encoded_Phase, Scaling; Cache::MultiStep_Model_Cache)
#     Evaluate_Jacobian_Step_One!(Cache, M, X, Index_List, Data, Encoded_Phase, Scaling)
#     BB_RS = Cache.Forward_Jac
#     Residual = Cache.Residual
#     NT = length(Index_List) - 1
#     HH22 = reshape(HH, size(Data, 1), NT, size(Data, 1), NT)
#     for t in 1:NT
#         R_range = 1 + Index_List[t] : Index_List[t+1]
#         Scaling_R = view(Scaling, R_range)
#         BB_RS_Rt = view(BB_RS, :, R_range, :)
#         # Hessian
#         HH22_R = view(HH22, :, t, :, t)
#         @tullio HH22_R[i, j] = BB_RS_Rt[p, k, i] * BB_RS_Rt[p, k, j] * Scaling_R[k]
#     end
#     return HH
# end
#
# function Optimise_IC_New!(M::MultiStep_Model, X, Index_List, Data, Encoded_Phase, Scaling; Cache::MultiStep_Model_Cache = Make_Cache(M, X, Index_List, Data, Encoded_Phase, Scaling), check=false, Radius = 0, Maximum_Radius = 4 * sqrt(manifold_dimension(M)), Iterations=32)
#     loss = x -> Loss(M, ArrayPartition(X.WW, reshape(x, size(X.IC)...)), Index_List, Data, Encoded_Phase, Scaling; Cache=Cache)
#     grad! = (g, x) -> Gradient_IC!(g, M, ArrayPartition(X.WW, reshape(x, size(X.IC)...)), Index_List, Data, Encoded_Phase, Scaling; Cache=Cache)
#     hess! = (h, x) -> Hessian_IC!(h, M, ArrayPartition(X.WW, reshape(x, size(X.IC)...)), Index_List, Data, Encoded_Phase, Scaling; Cache=Cache)
#     res = Optim.optimize(loss, grad!, hess!, vec(X.IC), method=NewtonTrustRegion())
#     return res
# end

function Test_Jacobian(M::MultiStep_Model, X, Index_List, Data, Encoded_Phase, JJR)
    Values = zeros(eltype(X), size(Data)...)
    Values_FD = zeros(eltype(X), size(Data)...)
    X_FD = deepcopy(X)
    Jac_FD = zeros(eltype(X), size(Data)..., length(X))
    Evaluate!(Values, M, X, Index_List, Data, Encoded_Phase)
    Eps = 1e-8
    for k in eachindex(X)
        X_FD[k] += Eps
        Evaluate!(Values_FD, M, X_FD, Index_List, Data, Encoded_Phase)
        Jac_FD[:, :, k] .= (Values_FD - Values) ./ Eps
        X_FD[k] = X[k]
        @show norm(JJR[:, :, k] - Jac_FD[:, :, k])
    end
    return Jac_FD
end

# Fills in the values for
#   Values,
#   Monomials, # The full set of Monomials, down to constant
#   Residual
# Creates empty containers for
#   Jac
#   Forward_Jac
#   BB_Jac
#   Hessian # HH
#   Gradient # GG
function Make_Cache(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Big_Jac = true,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    #     println("MultiStep_Model: Make_Cache")
    Library_Dimension = size(M.Monomial_Exponents, 2)
    Values = zero(Data)
    Monomials = zeros(eltype(X), Library_Dimension, size(Data, 2))
    Evaluate!(Values, Monomials, M, X, Index_List, Data, Encoded_Phase)
    Residual = deepcopy(Values)
    Residual .-= Data
    Jac = zeros(eltype(X), size(Data, 1), size(Data)...)
    Forward_Jac = zeros(eltype(X), size(Data)..., size(Data, 1))
    BB_Jac = Big_Jac ? zeros(eltype(X), size(Data)..., size(X.WW)...) : []
    @show size(BB_Jac)
    Hessian = zeros(eltype(X), length(X), length(X))# HH
    Gradient = zeros(eltype(X), length(X))
    Cache = MultiStep_Model_Cache{
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        Trajectories,
    }(
        Values,
        Monomials,
        Residual,
        Jac,
        Forward_Jac,
        BB_Jac,
        Hessian,
        Gradient,
    )
    return Cache
end

# Only update Values, Monomials and Residual
function Update_Cache!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
)
    #     println("MultiStep_Model: Update_Cache_Data!")
    Evaluate!(Cache.Values, Cache.Monomials, M, X, Index_List, Data, Encoded_Phase)
    Cache.Residual .= Cache.Values .- Data
    return nothing
end

# Only updates Residual
function Update_Cache_Data!(
    Cache::MultiStep_Model_Cache,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling,
)
    #     println("MultiStep_Model: Update_Cache_Data!")
    Cache.Residual .= Cache.Values .- Data
    return nothing
end

function Loss(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
)
    loss = sum(vec(sum(Cache.Residual .^ 2, dims = 1)) .* Scaling) / 2
    #     println("MultiStep_Model: Loss = ", loss)
    return loss
end

function Loss_With_Update(
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
)
    Evaluate!(Cache.Values, Cache.Monomials, M, X, Index_List, Data, Encoded_Phase)
    Cache.Residual .= Cache.Values .- Data
    loss = sum(vec(sum(Cache.Residual .^ 2, dims = 1)) .* Scaling) / 2
    println("MultiStep_Model: Loss = ", loss)
    return loss
end

# updates the Cache after each calculation
# Updates after optimisation
#   Values,
#   Monomials,
#   Residual
# Calculates before optimisation
#   Jac
#   Forward_Jac
#   BB_Jac
#   Hessian # HH
#   Gradient # GG

function Pointwise_Error!(
    Error,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
)
    Error .= sqrt.(vec(sum(Cache.Residual .^ 2, dims = 1)) .* Scaling)
    return nothing
end

function Pointwise_Gradient!(
    Gradient,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
)
    Residual = Cache.Residual
    @tullio Gradient[i, k] = -Residual[i, k] * Scaling[k]
    nothing
end

function Pointwise_Hessian!(
    Hessian,
    M::MultiStep_Model,
    X,
    Index_List,
    Data,
    Encoded_Phase,
    Scaling;
    Cache::MultiStep_Model_Cache = Make_Cache(
        M,
        X,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling,
    ),
)
    Id = Diagonal(ones(size(Data, 1)))
    @tullio Hessian[i, j, k] = Id[i, j] * Scaling[k]
    nothing
end

function Decompose(M::MultiStep_Model, Jacobian, Select)
    #     Jacobian = view(X.WW, :,:,M.Linear_Indices)
    SH = M.SH
    Vt, W, Lambda, Vt_C, W_C, Lambda_C =
        Decompose_Model_Right(Jacobian, SH; Time_Step = 1.0)
    @tullio W_C_SH[i, k, j] := W_C[i, l, j] * SH[l, k]
    return W_C[:, :, Select], Lambda_C[Select]
end

# Embeds the selected vector bundle
function Find_Tangent(M::MultiStep_Model, Jacobian, Select, Beta_Grid)
    WW, Lambda = Decompose(M, Jacobian, Select)
    Wr = real.(WW[:, :, 1])
    Wi = imag.(WW[:, :, 1])
    T0 = -angle(Lambda[1])
    #     T0 = angle(Lambda[1])
    DR0 = abs(Lambda[1])
    @tullio DW[i, j, k] := Wr[i, j] * cos(Beta_Grid[k]) + Wi[i, j] * sin(Beta_Grid[k])
    # check if this is really invariant
    #     Jacobian = view(X.WW, :,:,M.Linear_Indices)
    #     # F( W(r, beta, theta)) -> r = 1
    #     @tullio F_W[i, j, k] := Jacobian[i, j, p] * DW[p, j, k]  # j -> theta, k -> beta
    #     @tullio SH_beta[q, k] := psi(Beta_Grid[q] + T0 - Beta_Grid[k], length(Beta_Grid))
    #     @tullio W_RT[i, j, k] := DR0 * DW[i, l, q] * SH_beta[k, q] * SH[l, j]
    #     println("F_W")
    #     display(F_W)
    #     @show reshape(F_W, size(F_W, 1), :)
    #     @show W_RT
    #     @show F_W - W_RT
    #     @show DW
    return DW, DR0, T0
end

function Jacobian_Function_Helper!(Jac, X, Monomials, Encoded_Phase, D1row, D1val, D1col)
    for r in axes(D1row, 2),
        p in axes(D1row, 1),
        q in axes(Encoded_Phase, 1),
        i in axes(X, 1)

        a = (X[i, q, D1row[p, r]] * D1val[p, r])
        for k in axes(Encoded_Phase, 2)
            Jac[i, r, k] += a * Monomials[D1col[p, r], k] * Encoded_Phase[q, k]
        end
    end
    nothing
end

function Jacobian_Function_Helper_Auto!(Jac, X, Monomials, D1row, D1val, D1col)
    for r in axes(D1row, 2), p in axes(D1row, 1), i in axes(X, 1)

        a = (X[i, 1, D1row[p, r]] * D1val[p, r])
        for k in axes(Monomials, 2)
            Jac[i, r, k] += a * Monomials[D1col[p, r], k]
        end
    end
    nothing
end

function Jacobian_Function!(Jac, M::MultiStep_Model, X, Data, Encoded_Phase)
    Monomials = zeros(eltype(Data), size(M.Monomial_Exponents, 2), size(Data, 2))
    Evaluate_Library!(Monomials, M.Monomial_Exponents, Data)
    Jac .= 0
    if size(X.WW, 2) == 1
        Jacobian_Function_Helper_Auto!(Jac, X.WW, Monomials, M.D1row, M.D1val, M.D1col)
    else
        Jacobian_Function_Helper!(
            Jac,
            X.WW,
            Monomials,
            Encoded_Phase,
            M.D1row,
            M.D1val,
            M.D1col,
        )
    end
    return nothing
end

function Jacobian_Function_Test(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Data,
    Encoded_Phase,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    Jac = zeros(State_Dimension, State_Dimension, size(Data, 2))
    Value = zeros(State_Dimension, size(Data, 2))
    Evaluate_Function!(Value, M, X, Data, Encoded_Phase)
    Jacobian_Function!(Jac, M, X, Data, Encoded_Phase)
    Data_FD = deepcopy(Data)
    Value_FD = deepcopy(Value)
    Jac_FD = deepcopy(Jac)
    Eps = 1.0e-8
    for k = 1:State_Dimension
        Data_FD[k, :] .+= Eps
        Evaluate_Function!(Value_FD, M, X, Data_FD, Encoded_Phase)
        Jac_FD[:, k, :] .= (Value_FD - Value) ./ Eps
        Data_FD[k, :] .= Data[k, :]
        @show norm(Jac[:, k, :]), norm(Jac_FD[:, k, :] - Jac[:, k, :])
        @show Jac[:, k, 1:4]
        @show Jac_FD[:, k, 1:4]
    end
end

# cannot do periodic forcing! It has to be a periodic system with t2 as the [0,2 pi] parameter
# it can represent an ODE or a discrete-time map
function Model_From_Function(
    fun!,
    par,
    omega;
    State_Dimension,
    Skew_Dimension,
    Start_Order,
    End_Order,
)
    M_Model = MultiStep_Model(State_Dimension, Skew_Dimension, Start_Order, End_Order, 1)
    Monomial_Exponents = M_Model.Monomial_Exponents[:, M_Model.Admissible]
    XX = zero(M_Model)
    X_Model = XX.WW
    u0 = set_variables("x", numvars = State_Dimension, order = End_Order)
    grid = Fourier_Grid(Skew_Dimension)
    M_Model.SH .= Shift_Operator(grid, omega)
    for j in eachindex(grid)
        y = similar(u0)
        fun!(y, u0, par, 0, grid[j])
        for k = 1:State_Dimension
            for i in axes(Monomial_Exponents, 2)
                X_Model[k, j, i] = getcoeff(y[k], Monomial_Exponents[:, i])
            end
        end
    end
    return M_Model, XX
end

# Vectorfield!(x, y, Parameters, Alpha)
# Alpha_Generator(Parameters)
"""
    Model_From_Function_Alpha(
        Vectorfield!,
        Alpha_Generator,
        Parameters;
        State_Dimension,
        Start_Order,
        End_Order
    )

Create a polynomial vector field by Taylor expanding the ODE given by `Vectorfield!` and
`Alpha_Generator`.

It uses the same definition a a vector field as [`Generate_From_ODE`](@ref). The difference is that
`Alpha_Generator` is the infinitesimal generator matrix of the forcing dynamics.

`Start_Order` and `End_Order` set the representing polynomial orders.
"""
function Model_From_Function_Alpha(
    Vectorfield!,
    Forcing_Map!,
    Alpha_Generator,
    IC_Force,
    Parameters;
    State_Dimension,
    Start_Order,
    End_Order,
)
    Generator = Alpha_Generator(Parameters)
    Skew_Dimension = size(Generator, 1)
    M_Model = MultiStep_Model(State_Dimension, Skew_Dimension, Start_Order, End_Order, 1)
    Monomial_Exponents = M_Model.Monomial_Exponents[:, M_Model.Admissible]
    XX = zero(M_Model)
    X_Model = XX.WW
    u0 = set_variables("x", numvars = State_Dimension, order = End_Order)
    Alpha = zeros(eltype(Generator), Skew_Dimension)
    M_Model.SH .= Generator
    y = similar(u0)
    for j in eachindex(Alpha)
        Alpha[j] = 1
        Vectorfield!(y, u0, Forcing_Map!(IC_Force, Alpha, Parameters), Parameters)
        for k = 1:State_Dimension
            for i in axes(Monomial_Exponents, 2)
                X_Model[k, j, i] = getcoeff(y[k], Monomial_Exponents[:, i])
            end
        end
        Alpha[j] = 0
    end
    return M_Model, XX, Generator
end

function Find_Torus(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X;
    Iterations = 100,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    SH = M.SH
    grid = Fourier_Grid(Skew_Dimension)
    Torus = zeros(State_Dimension, Skew_Dimension)
    Torus_SH = zero(Torus)
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Residual = zero(Torus)
    Jac = zeros(State_Dimension, State_Dimension, Skew_Dimension)
    Id_Skew = Diagonal(ones(State_Dimension))
    Big_Jac = zeros(State_Dimension, Skew_Dimension, State_Dimension, Skew_Dimension)
    for k = 1:Iterations
        Evaluate_Function!(Residual, M, X, Torus, Encoded_Phase)
        @tullio Torus_SH[i, j] = Torus[i, p] * SH[p, j]
        Residual .-= Torus_SH
        Jacobian_Function!(Jac, M, X, Torus, Encoded_Phase)
        for p in axes(SH, 1), q in axes(SH, 2)
            #             @show size(Big_Jac[:, p, :, q]), size(Jac[:, :, p])
            Big_Jac[:, p, :, q] .= Jac[:, :, p] * I[p, q] - Id_Skew * SH[q, p]
        end
        Big_Jac_RS = reshape(
            Big_Jac,
            State_Dimension * Skew_Dimension,
            State_Dimension * Skew_Dimension,
        )
        #         @show minimum(abs.(eigvals(Big_Jac_RS)))
        Delta_RS = Big_Jac_RS \ vec(Residual)
        Torus .-= reshape(Delta_RS, State_Dimension, Skew_Dimension)
        #         @show norm(Residual), norm(Delta_RS), norm(Torus)
        if maximum(abs.(Residual)) < 16 * eps(eltype(Residual))
            Jacobian_Function!(Jac, M, X, Torus, Encoded_Phase)
            return Torus, permutedims(Jac, (1, 3, 2))
        end
    end
    return Torus, permutedims(Jac, (1, 3, 2))
end

function Find_Torus_ODE(
    M::MultiStep_Model{State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories},
    X,
    Generator;
    Iterations = 100,
) where {State_Dimension,Skew_Dimension,Start_Order,End_Order,Trajectories}
    grid = Fourier_Grid(Skew_Dimension)
    DD = differentialOperator(grid)
    Torus = zeros(State_Dimension, Skew_Dimension)
    Torus_SH = zero(Torus)
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Residual = zero(Torus)
    Jac = zeros(State_Dimension, State_Dimension, Skew_Dimension)
    Id_Skew = Diagonal(ones(State_Dimension))
    Big_Jac = zeros(State_Dimension, Skew_Dimension, State_Dimension, Skew_Dimension)
    for k = 1:Iterations
        Evaluate_Function!(Residual, M, X, Torus, Encoded_Phase)
        @tullio Residual[i, j] += -Torus[i, p] * Generator[j, p]
        Jacobian_Function!(Jac, M, X, Torus, Encoded_Phase)
        for p in axes(Big_Jac, 2), q in axes(Big_Jac, 4)
            #             @show size(Big_Jac[:, p, :, q]), size(Jac[:, :, p])
            Big_Jac[:, p, :, q] .= Jac[:, :, p] * I[p, q] - Id_Skew * Generator[p, q]
        end
        Big_Jac_RS = reshape(
            Big_Jac,
            State_Dimension * Skew_Dimension,
            State_Dimension * Skew_Dimension,
        )
        #         @show minimum(abs.(eigvals(Big_Jac_RS)))
        Delta_RS = Big_Jac_RS \ vec(Residual)
        Torus .-= reshape(Delta_RS, State_Dimension, Skew_Dimension)
        #         @show norm(Residual), norm(Delta_RS), norm(Torus)
        if maximum(abs.(Residual)) < 16 * eps(eltype(Residual))
            Jacobian_Function!(Jac, M, X, Torus, Encoded_Phase)
            return Torus, Jac
        end
    end
    return Torus, Jac
end

# function Model_About_Torus(M::MultiStep_Model{State_Dimension, Skew_Dimension, Start_Order, End_Order, Trajectories}, X, Omega_ODE; Iterations=100)
#     Torus = Find_Torus_ODE(M, X, Omega_ODE; Iterations=Iterations)
#     XDK = thetaDerivative(MK, XK, Omega_ODE)
# canMap = (x,t) -> Eval(MP, XP, t, Eval(MK, XK, t) + x) - Eval(MK, XDK, t)
# XP0 = fromFunction(MP, canMap)
# end

# moving the torus to the origin
# (x, y, p, t)
# Alpha : initial condition
# Alpha_Matrix : temporary storage
# Vectorfield!(x, y, p, Alpha(t)) : the right hand side
# Alpha_Map!(Alpha_Matrix, Parameters, t) : the transition matrix
function ode_with_jacobian!(
    dz_f,
    z_f,
    Parameters,
    t,
    Alpha,
    Alpha_Matrix,
    Vectorfield!,
    Forcing_Map!,
    Alpha_Map!,
    IC_Force,
)
    u = z_f.u
    jac = z_f.jac
    d_u = dz_f.u
    d_jac = dz_f.jac
    ForwardDiff.jacobian!(
        d_jac,
        (r, u) -> Vectorfield!(
            r,
            u,
            Forcing_Map!(
                IC_Force,
                Alpha_Map!(Alpha_Matrix, Parameters, t) * Alpha,
                Parameters,
            ),
            Parameters,
        ),
        d_u,
        u,
    )
    d_jac .= d_jac * jac
    return dz_f
end

function Time_Stepping_With_Jacobian!(
    Value,
    Jac,
    Vectorfield!,
    Forcing_Map!,
    Alpha_Map!,
    IC_Force,
    Parameters,
    u0, # state initial condition (an array)
    Alpha, # forcing initial condition (an array)
    Alpha_Matrix, # matrix storage
    Time_Step;
    abstol = 1e-10,
    reltol = 1e-10,
)
    Time_Span = (0, Time_Step)
    for k in axes(u0, 2)
        ODE_Problem = ODEProblem(
            (x, y, p, t) -> ode_with_jacobian!(
                x,
                y,
                p,
                t,
                Alpha[:, k],
                Alpha_Matrix,
                Vectorfield!,
                Forcing_Map!,
                Alpha_Map!,
                IC_Force,
            ),
            ComponentArray(u = u0[:, k], jac = diagm(ones(size(u0, 1)))),
            Time_Span,
            Parameters,
        )
        Solution = solve(ODE_Problem, Feagin14(), abstol = abstol, reltol = reltol)
        res = Solution(Time_Step)
        Value[:, k] .= res.u
        Jac[:, :, k] .= res.jac
    end
    # display(Value)
    # display(Jac)
    return nothing
end

# using shooting method
function Torus_From_ODE(
    Vectorfield!,
    Forcing_Map!,
    Alpha_Map!,
    IC_Force,
    Parameters,
    Time_Step;
    State_Dimension,
    Skew_Dimension,
    Iterations = 100,
    abstol = 1e-10,
    reltol = 1e-10,
)
    Alpha_Matrix = zeros(Skew_Dimension, Skew_Dimension)
    SH = zeros(Skew_Dimension, Skew_Dimension)
    Alpha_Map!(SH, Parameters, Time_Step)
    Torus = zeros(State_Dimension, Skew_Dimension)
    Torus_SH = zero(Torus)
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Residual = zero(Torus)
    Jac = zeros(State_Dimension, State_Dimension, Skew_Dimension)
    Id_State = Diagonal(ones(State_Dimension))
    Id_Skew = Diagonal(ones(Skew_Dimension))
    Big_Jac = zeros(State_Dimension, Skew_Dimension, State_Dimension, Skew_Dimension)
    for k = 1:Iterations
        Time_Stepping_With_Jacobian!(
            Residual,
            Jac,
            Vectorfield!,
            Forcing_Map!,
            Alpha_Map!,
            IC_Force,
            Parameters,
            Torus, # state initial condition (an array)
            Id_Skew, # forcing initial condition (an array)
            Alpha_Matrix, # matrix storage
            Time_Step;
            abstol = abstol,
            reltol = reltol,
        )
        @tullio Torus_SH[i, j] = Torus[i, p] * SH[p, j]
        Residual .-= Torus_SH
        for p in axes(SH, 1), q in axes(SH, 2)
            Big_Jac[:, p, :, q] .= Jac[:, :, p] * I[p, q] - Id_State * SH[q, p]
        end
        Big_Jac_RS = reshape(
            Big_Jac,
            State_Dimension * Skew_Dimension,
            State_Dimension * Skew_Dimension,
        )
        Delta_RS = Big_Jac_RS \ vec(Residual)
        Torus .-= reshape(Delta_RS, State_Dimension, Skew_Dimension)
        @show norm(Residual), norm(Delta_RS), norm(Torus)
        if maximum(abs.(Residual)) < 16 * eps(eltype(Residual))
            Time_Stepping_With_Jacobian!(
                Residual,
                Jac,
                Vectorfield!,
                Forcing_Map!,
                Alpha_Map!,
                IC_Force,
                Parameters,
                Torus, # state initial condition (an array)
                Id_Skew, # forcing initial condition (an array)
                Alpha_Matrix, # matrix storage
                Time_Step;
                abstol = abstol,
                reltol = reltol,
            )
            return Torus, Jac
        end
    end
    return Torus, Jac
end

"""
    Model_From_ODE(
        Vectorfield!,
        Forcing_Map!,
        Alpha_Map!,
        IC_Force,
        Parameters,
        dt,
        Time_Step;
        State_Dimension,
        Skew_Dimension,
        Start_Order,
        End_Order,
        Steady_State::Bool = true,
        Iterations = 100,
    )

Creates a discrete-time model from the vector field
`Vectorfield!`, `Forcing_Map!` and `Alpha_Map!`. The definition is the same as in [`Generate_From_ODE`](@ref).
The routine first calculates a steady state of the system straight from the ODE.
Then it create a polynomial map about the steady state. The resulting map has its steady state at the origin.

* `dt` time step of the ODE solver
* `Time_Step` integration time of the ODE. One iteration of the resulting map is the same as solveing the ODE for `Time_Step` time.
* `State_Dimension` state space dimension
* `Skew_Dimension` forcing space dimension
* `Start_Order` starting order of the resulting map
* `End_Order` highest order monomial in the resulting map
* `Iterations` how many Newto iteration to take when solving for the steady state.

"""
function Model_From_ODE(
    Vectorfield!,
    Forcing_Map!,
    Alpha_Map!,
    IC_Force,
    Parameters,
    dt,
    Time_Step;
    State_Dimension,
    Skew_Dimension,
    Start_Order,
    End_Order,
    Steady_State::Bool = true,
    Iterations = 100,
)
    #     Torus = ifelse(Steady_State,
    #                    zeros(State_Dimension, Skew_Dimension),
    #                    Torus_From_ODE(fun, par, Omega_ODE, Time_Step; State_Dimension=State_Dimension, Skew_Dimension=Skew_Dimension, Iterations=Iterations)[1])
    Alpha_Matrix = zeros(Skew_Dimension, Skew_Dimension)
    SH = zeros(Skew_Dimension, Skew_Dimension)
    Alpha_Map!(SH, Parameters, Time_Step)
    Torus, _ = Torus_From_ODE(
        Vectorfield!,
        Forcing_Map!,
        Alpha_Map!,
        IC_Force,
        Parameters,
        Time_Step;
        State_Dimension = State_Dimension,
        Skew_Dimension = Skew_Dimension,
        Iterations = Iterations,
    )
    #     Torus = zeros(State_Dimension, Skew_Dimension)
    # @show Torus
    Id_Skew = Diagonal(ones(Skew_Dimension))
    M_Model = MultiStep_Model(State_Dimension, Skew_Dimension, Start_Order, End_Order, 1)
    Monomial_Exponents = M_Model.Monomial_Exponents[:, M_Model.Admissible]
    XX = zero(M_Model)
    X_Model = XX.WW
    u0 = set_variables("x", numvars = State_Dimension, order = End_Order)
    grid = Fourier_Grid(Skew_Dimension)
    M_Model.SH .= SH
    for j in eachindex(grid)
        tspan = (0, Time_Step)
        ODE_Problem = ODEProblem(
            (x, y, p, t) -> Vectorfield!(
                x,
                y,
                Forcing_Map!(IC_Force, Alpha_Map!(Alpha_Matrix, p, t) * Id_Skew[:, j], p),
                p,
            ),
            Torus[:, j] + u0,
            tspan,
            Parameters,
        )
        sol = solve(
            ODE_Problem,
            Tsit5(),
            dt = dt,
            internalnorm = (u, t) -> 0.0,
            adaptive = false,
        )
        y = sol[end]
        for k = 1:State_Dimension
            for i in axes(Monomial_Exponents, 2)
                X_Model[k, j, i] = getcoeff(y[k], Monomial_Exponents[:, i])
            end
        end
    end
    X_Model[:, :, M_Model.Constant_Indices[1]] .-= Torus * SH
    return M_Model, XX
end
