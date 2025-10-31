
## ---------------------------------------------------------------------------------------
## TensorManifold
##
## ---------------------------------------------------------------------------------------

struct TensorManifold{𝔽} <: AbstractDecoratorManifold{𝔽}
    ranks::Array{T,1} where {T<:Integer}
    children::Array{T,2} where {T<:Integer}
    dim2ind::Array{T,1} where {T<:Integer}
    manifold::ProductManifold
end

function Base.zero(M::TensorManifold{field}) where {field}
    return ArrayPartition(map(zero, M.manifold.manifolds))
end

@inline Base.getindex(M::TensorManifold, i::Integer) = M.manifold[i]

@inline ManifoldsBase.decorated_manifold(M::TensorManifold) = M.manifold
@inline ManifoldsBase.get_forwarding_type(::TensorManifold, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::TensorManifold, ::Any, _) = ManifoldsBase.SimpleForwardingType()

# internal
function nr_nodes(children::Array{T,2}) where {T<:Integer}
    return size(children, 1)
end

function nr_nodes(hten::TensorManifold{𝔽}) where {𝔽}
    return nr_nodes(hten.children)
end

function is_leaf(children::Array{T,2}, ii) where {T<:Integer}
    return prod(children[ii, :] .== 0)
end

function is_leaf(hten::TensorManifold{𝔽}, ii) where {𝔽}
    return is_leaf(hten.children, ii)
end

# import Base.size
#
# function size(M::TensorManifold{𝔽}) where T
#     return Tuple(size(X.x[k],1) for k in M.dim2ind)
# end

# define_tree(d, tree_type = :balanced)
# creates a tree structure from dimensions 'd'
function define_tree(d, tree_type = :balanced)
    children = zeros(typeof(d), 2 * d - 1, 2)
    dims = [collect(1:d)]

    nr_nodes = 1
    ii = 1
    while ii <= nr_nodes
        if length(dims[ii]) == 1
            children[ii, :] = [0, 0]
        else
            ii_left = nr_nodes + 1
            ii_right = nr_nodes + 2
            nr_nodes = nr_nodes + 2
            push!(dims, [])
            push!(dims, [])

            children[ii, :] = [ii_left, ii_right]
            if tree_type == :first_separate && ii == 1
                dims[ii_left] = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            elseif tree_type == :first_pair_separate && ii == 1 && d > 2
                dims[ii_left] = dims[ii][1:2]
                dims[ii_right] = dims[ii][3:end]
            elseif tree_type == :TT
                dims[ii_left] = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            else
                dims[ii_left] = dims[ii][1:div(end, 2)]
                dims[ii_right] = dims[ii][(div(end, 2)+1):end]
            end
        end
        ii += 1
    end
    ind_leaves = findall(children[:, 1] .== 0)
    pivot = [dims[k][1] for k in ind_leaves]
    dim2ind = zero(pivot)
    dim2ind[pivot] = ind_leaves
    return children, dim2ind
end

# a complicated constructor
function TensorManifold(
    dims::Array{T,1},
    ranks::Array{T,1},
    children,
    dim2ind,
    tree_type = :balanced;
    field::AbstractNumbers = ℝ,
) where {T<:Integer}
    M = []
    for ii = 1:nr_nodes(children)
        if is_leaf(children, ii)
            dim_id = findfirst(isequal(ii), dim2ind)
            n_ii = dims[dim_id]
            @assert n_ii >= ranks[ii] "Mismatched ranks."
            push!(M, Stiefel(n_ii, ranks[ii]))
        else
            ii_left = children[ii, 1]
            ii_right = children[ii, 2]
            if ii == 1
                push!(M, Euclidean(ranks[ii_left] * ranks[ii_right], ranks[ii]))
            else
                @assert ranks[ii_left] * ranks[ii_right] >= ranks[ii] "Mismatched ranks."
                push!(M, Stiefel(ranks[ii_left] * ranks[ii_right], ranks[ii]))
            end
        end
    end
    return TensorManifold{field}(ranks, children, dim2ind, ProductManifold(M...))
end

# create a rank structure such that 'ratio' is the lost rank
function cascade_ranks(
    children,
    dim2ind,
    topdim,
    dims;
    node_ratio = 1.0,
    leaf_ranks = [max(1.0, d / 2) for d in dims],
    max_rank = 16,
)
    ranks = zero(children[:, 1])
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            id = findfirst(isequal(ii), dim2ind)
            n_ii = dims[id]
            ranks[ii] = max(1.0, round(leaf_ranks[id]))
            if ranks[ii] > min(n_ii, max_rank)
                ranks[ii] = min(n_ii, max_rank)
            end
        else
            ii_left = children[ii, 1]
            ii_right = children[ii, 2]
            ranks[ii] = round(ranks[ii_left] * ranks[ii_right] * node_ratio)
            if ranks[ii] > min(ranks[ii_left] * ranks[ii_right], max_rank)
                ranks[ii] = min(ranks[ii_left] * ranks[ii_right], max_rank)
            end
        end
    end
    return ranks
end

# #  the output rank cannot be greater than the input rank
# function prune_ranks!(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind) where {T<:Integer}
#     ranks[1] = topdim
#     for ii = nr_nodes(children):-1:2
#         if is_leaf(children, ii)
#             n_ii = dims[findfirst(isequal(ii), dim2ind)]
#             if ranks[ii] > n_ii
#                 ranks[ii] = n_ii
#             end
#         else
#             ii_left = children[ii, 1]
#             ii_right = children[ii, 2]
#             if ranks[ii] > ranks[ii_left] * ranks[ii_right]
#                 ranks[ii] = ranks[ii_left] * ranks[ii_right]
#             end
#         end
#     end
#     return nothing
# end

# TODO: documentation
function HTTensor(
    dims::Array{T,1},
    topdim::T = 1,
    tree_type = :balanced;
    node_ratio = 1.0,
    leaf_ranks = [max(1.0, d / 2) for d in dims],
    max_rank = 16,
) where {T<:Integer}
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = cascade_ranks(
        children,
        dim2ind,
        topdim,
        dims,
        node_ratio = node_ratio,
        leaf_ranks = leaf_ranks,
        max_rank = max_rank,
    )
    @show ranks
    return TensorManifold(dims, ranks, children, dim2ind, tree_type)
end

# # normally we don't need this
# function getel(hten::TensorManifold{field}, X, idx) where {field}
#     # this goes from the top level to the lowest level
#     Forward_Product = [zeros(Float64, 0) for k = 1:size(hten.children, 1)]
#     for ii = size(hten.children, 1):-1:1
#         if is_leaf(hten, ii)
#             sel = idx[findfirst(isequal(ii), hten.dim2ind)]
#             Forward_Product[ii] = X.x[ii][sel, :]
#         else
#             ii_left = hten.children[ii, 1]
#             ii_right = hten.children[ii, 2]
#             s_l = size(X.x[ii_left], 2)
#             s_r = size(X.x[ii_right], 2)
#             Forward_Product[ii] = [transpose(Forward_Product[ii_left]) * reshape(X.x[ii][:, k], s_l, s_r) * Forward_Product[ii_right] for k = 1:size(X.x[ii], 2)]
#         end
#     end
#     return Forward_Product[1][idx[end]]
# end
#
# # normally we don't need this
# """
#     Calculates the Gramian matrices of the HT tensor
# """
# function gramians(hten::TensorManifold{field}, X) where {field}
#     gr = Array{Array{Float64,2},1}(undef, size(hten.children, 1))
#     gr[1] = transpose(X.x[1]) * X.x[1]
#     for ii = 1:size(hten.children, 1)
#         if !is_leaf(hten, ii)
#             # Child nodes
#             ii_left = hten.children[ii, 1]
#             ii_right = hten.children[ii, 2]
#             s_l = size(X.x[ii_left], 2)
#             s_r = size(X.x[ii_right], 2)
#
#             #             % Calculate contractions < B{ii}, G{ii} o_1 B{ii} >_(1, 2) and _(1, 3)
#             #             B_mod = ttm(conj(x.B{ii}), G{ii}, 3);
#             #             @show size(X.x[ii]), size(gr[ii])
#             #             @show size(reshape(X.x[ii], s_l, s_r,:)), size(reshape(gr[ii], size(gr[ii],1), size(gr[ii],2), 1))
#             B_mod = reshape(X.x[ii], s_l, s_r, :) * gr[ii]
#             #   G{ii_left } = ttt(conj(x.B{ii}), B_mod, [2 3], [2 3], 1, 1);
#             #   G{ii_right} = ttt(conj(x.B{ii}), B_mod, [1 3], [1 3], 2, 2);
#             gr[ii_left] = dropdims(sum(reshape(X.x[ii], s_l, 1, s_r, :) .* reshape(B_mod, 1, s_l, s_r, :), dims=(3, 4)), dims=(3, 4))
#             gr[ii_right] = dropdims(sum(reshape(X.x[ii], s_l, 1, s_r, :) .* reshape(B_mod, s_l, s_r, 1, :), dims=(1, 4)), dims=(1, 4))
#         end
#     end
#     return gr
# end
#
# # normally we don't need this
# # orthogonalise the non-root nodes
# """
#     This orthogonalises the non-root nodes of the HT tensor, by leaving the value unchanged.
#     Normally, this method is not needed, because our algorithms keep these matrices orthogonal
# """
# function orthog!(M::TensorManifold{field}, X) where {field}
#     for ii = nr_nodes(M):-1:2
#         if is_leaf(M, ii)
#             F = qr(X.x[ii])
#             X.x[ii] .= Array(F.Q)
#             R = F.R
#         else
#             F = qr(X.x[ii])
#             X.x[ii] .= Array(F.Q)
#             R = F.R
#         end
#         left_par = findfirst(isequal(ii), M.children[:, 1])
#         right_par = findfirst(isequal(ii), M.children[:, 2])
#         if left_par != nothing
#             for k = 1:size(X.x[left_par], 2)
#                 X.x[left_par][:, k] .= vec(R * reshape(X.x[left_par][:, k], size(R, 2), :))
#             end
#         else
#             for k = 1:size(X.x[right_par], 2)
#                 X.x[right_par][:, k] .= vec(reshape(X.x[right_par][:, k], :, size(R, 2)) * transpose(R))
#             end
#         end
#     end
#     return nothing
# end

# -------------------------------------------------------------------------------------
#
# In this section we calculate the hmtensor derivative of X with respect to X.U, X.B
# The Data is an array for multiple evaluations at the same time
# we need to allow for a missing index, so that a vector is output
# therefore we need to propagate a matrix through
#
# -------------------------------------------------------------------------------------

# this acts as a cache for the tensor evaluations and gradients
struct Tensor_Cache{T}
    Is_Valid_Forward::Array{Bool,1}
    Is_Valid_Reverse::Array{Bool,1}
    Forward_Product::Array{Array{T,3},1}
    Reverse_Product::Array{Array{T,3},1}
end

function Invalidate_Forward(DV::Tensor_Cache)
    DV.Is_Valid_Forward .= false
    nothing
end

function Invalidate_Reverse(DV::Tensor_Cache)
    DV.Is_Valid_Reverse .= false
    nothing
end

function Invalidate_All(DV::Tensor_Cache)
    DV.Is_Valid_Forward .= false
    DV.Is_Valid_Reverse .= false
    nothing
end

function Tensor_Cache(T, nodes)
    return Tensor_Cache{T}(
        zeros(Bool, nodes),
        zeros(Bool, nodes),
        Array{Array{T,3},1}(undef, nodes),
        Array{Array{T,3},1}(undef, nodes),
    )
end

function LmulBmulR!(vecs_ii, vecs_left, B, vecs_right)
    C = reshape(B, size(vecs_left, 1), size(vecs_right, 1), :)
    if size(vecs_left, 2) == 1
        @tullio vecs_ii[jb, jr, k] = vecs_left[p, 1, k] * C[p, q, jb] * vecs_right[q, jr, k]
    elseif size(vecs_right, 2) == 1
        @tullio vecs_ii[jb, jl, k] = vecs_left[p, jl, k] * C[p, q, jb] * vecs_right[q, 1, k]
    end
    return nothing
end

function BmulData!(vecs_ii, B, data_wdim)
    @tullio vecs_ii[p, 1, k] = B[q, p] * data_wdim[q, k]
    return nothing
end

# contracts Data[i] with index[i] of the tensor X
# if i > length(Data) Data[end] is contracted with index[i]
# if third != 0, index[third] is not contracted
# output:
#   Forward_Product is a matrix for each Data point
# input:
#   X: is the tensor
#   Data: is a vector of two dimensional arrays.
#       The first dimension is the vector, the second dimension is the Data index
#   ii: the node to contract at
#   second: if non-zero Data[2] is contracted with this index, Data[1] with the rest except 'third'
#   third: if non-zero, the index not to contract
#   dt: to replace X at index rep
#   rep: where to replace X with dt
function Update_Forward_Product!(
    DV::Tensor_Cache{T},
    M::TensorManifold{field},
    X,
    Data...;
    ii,
) where {field,T}
    #     print("v=", ii, "/", length(X.x), "_")
    if DV.Is_Valid_Forward[ii]
        return nothing
    end
    B = X.x[ii]
    if is_leaf(M, ii)
        # it is a leaf
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(Data)
            wdim = length(Data)
        else
            wdim = dim
        end
        if ~isassigned(DV.Forward_Product, ii)
            DV.Forward_Product[ii] = zeros(T, size(B, 2), 1, size(Data[1], 2)) # Array{T}(undef, (size(B,2), 1, size(Data[1],2)))
        end
        #         @show size(DV.Forward_Product[ii]), size(B), size(Data[wdim])
        BmulData!(DV.Forward_Product[ii], B, Data[wdim])
    else
        # it is a node
        ii_left = M.children[ii, 1]
        ii_right = M.children[ii, 2]
        Update_Forward_Product!(DV, M, X, Data..., ii = ii_left)
        Update_Forward_Product!(DV, M, X, Data..., ii = ii_right)
        vs_l = size(DV.Forward_Product[ii_left], 2)
        vs_r = size(DV.Forward_Product[ii_right], 2)
        vs = max(vs_l, vs_r)
        if ~isassigned(DV.Forward_Product, ii)
            DV.Forward_Product[ii] = zeros(size(B, 2), vs, size(Data[1], 2))
        end

        LmulBmulR!(
            DV.Forward_Product[ii],
            DV.Forward_Product[ii_left],
            B,
            DV.Forward_Product[ii_right],
        )
    end
    DV.Is_Valid_Forward[ii] = true
    return nothing
end

# invalidates Forward_Product that are dependent on node "ii"
function Invalidate_Forward(
    DV::Tensor_Cache{T},
    M::TensorManifold{field},
    ii,
) where {field,T}
    DV.Is_Valid_Forward[ii] = false
    # not the root node
    if ii != 1
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:, 1])
        right = findfirst(isequal(ii), M.children[:, 2])
        if left != nothing
            Invalidate_Forward(DV, M, left)
        end
        if right != nothing
            Invalidate_Forward(DV, M, right)
        end
    end
    return nothing
end

# create partial results at the end of each node or leaf when multiplied with 'Data'
function Make_Forward_Product(M::TensorManifold{field}, X, Data...) where {field}
    DV = Tensor_Cache(eltype(Data[end]), size(M.children, 1))
    Update_Forward_Product!(DV, M, X, Data..., ii = 1)
    return DV
end

function Evaluate!(
    Result,
    M::TensorManifold{field},
    X,
    Data...;
    Cache::Tensor_Cache = Make_Forward_Product(M, X, Data...),
) where {field}
    Result .= dropdims(Cache.Forward_Product[1], dims = 2)
    #     @show size(Cache.Forward_Product)
    #     @show size(Cache.Forward_Product[1])
    #     Result .= Cache.Forward_Product[1]
    return nothing
end

# adds to the Result
function Evaluate_Add!(
    Result,
    M::TensorManifold{field},
    X,
    Data...;
    Cache::Tensor_Cache = Make_Forward_Product(M, X, Data...),
    Lambda = 1,
) where {field}
    #     Result .+= dropdims(Cache.Forward_Product[1], dims=2)
    res = Cache.Forward_Product[1]
    if Lambda == 1
        @tullio Result[i, j] += res[i, 1, j]
    else
        @tullio Result[i, j] += Lambda * res[i, 1, j]
    end
    return nothing
end

function BVpmulBmulV!(
    bvecs_ii::AbstractArray{T,3},
    B::AbstractArray{U,2},
    vecs_sibling::AbstractArray{T,3},
    bvecs_parent::AbstractArray{T,3},
) where {T,U}
    C = reshape(B, size(bvecs_ii, 2), size(vecs_sibling, 1), :)
    @tullio bvecs_ii[l, p, k] = C[p, q, r] * bvecs_parent[l, r, k] * vecs_sibling[q, 1, k]
    return nothing
end

function VmulBmulBVp!(
    bvecs_ii::AbstractArray{T,3},
    vecs_sibling::AbstractArray{T,3},
    B::AbstractArray{U,2},
    bvecs_parent::AbstractArray{T,3},
) where {T,U}
    C = reshape(B, size(vecs_sibling, 1), size(bvecs_ii, 2), :)
    @tullio bvecs_ii[l, q, k] = C[p, q, r] * bvecs_parent[l, r, k] * vecs_sibling[p, 1, k]
    return nothing
end

# Only supports full contraction
# dt is used for second derivatives. dt replaces X at node [rep].
# This is the same as taking the derivative w.r.t. node [rep] and multiplying by dt[rep]
# L0 is used to contract with tensor output
function Update_Reverse_Product_Parts!(
    DV::Tensor_Cache{T},
    M::TensorManifold{field},
    X;
    ii,
) where {field,T}
    #     print("b=", ii, "/", length(X.x), "_")
    # find the parent and multiply with the Forward_Product from the other brancs and the becs fron the bottom
    if DV.Is_Valid_Reverse[ii]
        return nothing
    end
    datalen = size(DV.Forward_Product[1], 3)
    if ii == 1
        DV.Reverse_Product[ii] = zeros(T, size(X.x[ii], 2), size(X.x[ii], 2), datalen)
        for k = 1:size(X.x[ii], 2)
            DV.Reverse_Product[ii][k, k, :] .= 1
        end
    else
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:, 1])
        right = findfirst(isequal(ii), M.children[:, 2])
        if left != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = Forward_Product[sibling]
            parent = left
            B = X.x[parent]
            s_l = size(X.x[M.children[parent, 1]], 2)
            s_r = size(X.x[M.children[parent, 2]], 2)
            sibling = M.children[left, 2] # right sibling
            Update_Reverse_Product_Parts!(DV, M, X, ii = parent)
            if ~isassigned(DV.Reverse_Product, ii)
                DV.Reverse_Product[ii] =
                    zeros(T, size(DV.Reverse_Product[parent], 1), s_l, datalen)
            end
            BVpmulBmulV!(
                DV.Reverse_Product[ii],
                B,
                DV.Forward_Product[sibling],
                DV.Reverse_Product[parent],
            )
        end
        if right != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = Forward_Product[sibling]
            parent = right
            B = X.x[parent]
            s_l = size(X.x[M.children[parent, 1]], 2)
            s_r = size(X.x[M.children[parent, 2]], 2)

            sibling = M.children[right, 1] # right sibling
            Update_Reverse_Product_Parts!(DV, M, X, ii = parent)
            #             @show size(X.B[parent]), size(DV.Reverse_Product[parent]), size(vec(Forward_Product[sibling]))
            if ~isassigned(DV.Reverse_Product, ii)
                DV.Reverse_Product[ii] =
                    zeros(T, size(DV.Reverse_Product[parent], 1), s_r, datalen)
            end
            VmulBmulBVp!(
                DV.Reverse_Product[ii],
                DV.Forward_Product[sibling],
                B,
                DV.Reverse_Product[parent],
            )
        end
    end
    DV.Is_Valid_Reverse[ii] = true
    return nothing
end

# if ii > 0  -> all children nodes get invalidate
function Invalidate_Reverse(
    DV::Tensor_Cache{T},
    M::TensorManifold{field},
    ii,
) where {field,T}
    # the root is always the identity, no need to update
    if is_leaf(M, ii)
        # do nothing as it has no children
    else
        ii_left = M.children[ii, 1]
        ii_right = M.children[ii, 2]
        DV.Is_Valid_Reverse[ii_left] = false
        DV.Is_Valid_Reverse[ii_right] = false
        Invalidate_Reverse(DV, M, ii_left)
        Invalidate_Reverse(DV, M, ii_right)
    end
    return nothing
end

function Update_Reverse_Product!(
    DV::Tensor_Cache,
    M::TensorManifold{field},
    X,
) where {field}
    if size(DV.Forward_Product[1], 2) != 1
        println("only supports full contraction")
        return nothing
    end
    datalen = size(DV.Forward_Product[1], 3)
    for ii = 1:nr_nodes(M)
        Update_Reverse_Product_Parts!(DV, M, X, ii = ii)
    end
    return nothing
end

function Make_Cache(M::TensorManifold, X, Data...)
    DV = Tensor_Cache(eltype(Data[end]), size(M.children, 1))
    Update_Forward_Product!(DV, M, X, Data...; ii = 1)
    Update_Reverse_Product!(DV, M, X)
    return DV
end

# update the content which is invalidated
function Update_Cache!(
    DV::Tensor_Cache,
    M::TensorManifold,
    X,
    Data::Vararg{AbstractMatrix{T}},
) where {T}
    DV.Is_Valid_Forward .= false
    DV.Is_Valid_Reverse .= false
    Update_Forward_Product!(DV, M, X, Data..., ii = 1)
    Update_Reverse_Product!(DV, M, X)
    return nothing
end

function Update_Cache_Parts!(
    DV::Tensor_Cache,
    M::TensorManifold,
    X,
    Data::Vararg{AbstractMatrix{T}};
    ii,
) where {T}
    Invalidate_Forward(DV, M, ii)
    Invalidate_Reverse(DV, M, ii)
    # make sure to update all the marked components
    # Vecs can start at the root (ii = 1), hence it covers the full tree
    Update_Forward_Product!(DV, M, X, Data..., ii = 1)
    # BVecs has to start from all the leaves to cover the whole tree
    Update_Reverse_Product!(DV, M, X)
    return nothing
end

# same as wDF, except that it only applies to node ii
function L0_DF_parts!(
    DF,
    M::TensorManifold{field},
    X,
    Data::Vararg{AbstractMatrix{T}};
    L0,
    ii::Integer = -1,
    Cache = Make_Cache(M, X, Data...),
) where {field,T}
    #     t0 = time()
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(Data)
            wdim = length(Data)
        else
            wdim = dim
        end
        dd = Data[wdim]
        bv = Cache.Reverse_Product[ii]
        @tullio DF[p, q] = L0[l, k+0] * dd[p, k] * bv[l, q, k]
    else
        # it is a node
        ch1 = Cache.Forward_Product[M.children[ii, 1]]
        ch2 = Cache.Forward_Product[M.children[ii, 2]]
        bv = Cache.Reverse_Product[ii]
        XOp = reshape(DF, size(ch1, 1), size(ch2, 1), size(DF, 2))
        #         @show size(L0), size(bv), size(XOp)
        @tullio XOp[p, q, r] = L0[l, k+0] * ch1[p, 1, k] * ch2[q, 1, k] * bv[l, r, k]
    end
    #     t1 = time()
    #     println("\n -> L0_DF = ", 100*(t1-t0))
    return nothing
end

function Hessian_Helper!(HH_RS, Scaling, ch1, ch2, bv)
    @inbounds @fastmath for k in eachindex(Scaling)
        for p1 in axes(ch1, 1), p2 in axes(ch1, 1), q1 in axes(ch2, 1), q2 in axes(ch2, 1)
            for r1 in axes(bv, 2), r2 in axes(bv, 2)
                a = zero(eltype(bv))
                for l in axes(bv, 1)
                    a += bv[l, r1, k] * bv[l, r2, k]
                end
                HH_RS[p1, q1, r1, p2, q2, r2] +=
                    a *
                    Scaling[k] *
                    ch1[p1, 1, k] *
                    ch2[q1, 1, k] *
                    ch1[p2, 1, k] *
                    ch2[q2, 1, k]
            end
        end
    end
    return nothing
end

# function Scaled_Hessian_parts!(HH, M::TensorManifold{field}, X, Data::Vararg{AbstractMatrix{T}}; Scaling, ii::Integer=-1, Cache=Make_Cache(M, X, Data...)) where {field,T}
#     #     t0 = time()
#     if is_leaf(M, ii)
#         dim = findfirst(isequal(ii), M.dim2ind)
#         if dim > length(Data)
#             wdim = length(Data)
#         else
#             wdim = dim
#         end
#         dd = Data[wdim]
#         bv = Cache.Reverse_Product[ii]
# #         @tullio DF[p2, q2] = L0[l1, l2, k+0] * dd[p1, k] * bv[l1, q1, k] * Delta[p1, q1] * dd[p2, k] * bv[l2, q2, k]
#         @tullio HH[p1, q1, p2, q2] = Scaling[k] * dd[p1, k] * bv[l, q1, k] * dd[p2, k] * bv[l, q2, k]
#     else
#         s_l = size(X.x[M.children[ii, 1]], 2)
#         s_r = size(X.x[M.children[ii, 2]], 2)
# #         Delta3D = reshape(Delta, s_l, s_r, :)
#         # it is a node
#         ch1 = Cache.Forward_Product[M.children[ii, 1]]
#         ch2 = Cache.Forward_Product[M.children[ii, 2]]
#         bv = Cache.Reverse_Product[ii]
# #         XOp = reshape(DF, size(ch1, 1), size(ch2, 1), size(DF, 2))
# #         @show size(L0), size(bv), size(XOp)
# #         XOp = reshape(DF, size(ch1, 1), size(ch2, 1), size(DF, 2))
# #         @tullio XOp[p2, q2, r2] = (L0[l1, l2, k] * ch1[p1, 1, k] * ch2[q1, 1, k] * bv[l1, r1, k] * Delta3D[p1, q1, r1]
# #                                     * ch1[p2, 1, k] * ch2[q2, 1, k] * bv[l2, r2, k])
#         HH_RS = reshape(HH, size(ch1, 1), size(ch2, 1), size(HH, 2), size(ch1, 1), size(ch2, 1), size(HH, 2))
#         if ii == 1
#             HH_VV = view(HH_RS, :, :, 1, :, :, 1)
#             @tullio HH_VV[p1, q1, p2, q2] = (Scaling[k] * ch1[p1, 1, k] * ch2[q1, 1, k] * ch1[p2, 1, k] * ch2[q2, 1, k])
#             for k in 2:size(HH_RS, 3)
#                 HH_RS[:, :, k, :, :, k] .= HH_RS[:, :, 1, :, :, 1]
#             end
#         else
#             Hessian_Helper!(HH_RS, Scaling, ch1, ch2, bv)
# #             @tullio HH_RS[p1, q1, r1, p2, q2, r2] = (Scaling[k] * ch1[p1, 1, k] * ch2[q1, 1, k] * bv[l, r1, k] * ch1[p2, 1, k] * ch2[q2, 1, k] * bv[l, r2, k])
#         end
#     end
#     #     t1 = time()
#     #     println("\n -> L0_DF = ", 100*(t1-t0))
#     return nothing
# end
# this multiplies the matrix valued L0 from the right side using
function L0_DF_DF_Delta_parts!(
    DF,
    Delta,
    Latent_Delta,
    M::TensorManifold{field},
    X,
    Data::Vararg{AbstractMatrix{T}};
    Scaling,
    ii::Integer = -1,
    Cache = Make_Cache(M, X, Data...),
) where {field,T}
    #     t0 = time()
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(Data)
            wdim = length(Data)
        else
            wdim = dim
        end
        dd = Data[wdim]
        bv = Cache.Reverse_Product[ii]
        # @tullio DF[p, q] = L0[l, k+0] * dd[p, k] * bv[l, q, k]
        @tullio Latent_Delta[l, k] = Scaling[k] * dd[p1, k] * bv[l, q1, k] * Delta[p1, q1]
        @tullio DF[p2, q2] = Latent_Delta[l, k] * dd[p2, k] * bv[l, q2, k]
    else
        s_l = size(X.x[M.children[ii, 1]], 2)
        s_r = size(X.x[M.children[ii, 2]], 2)
        Delta3D = reshape(Delta, s_l, s_r, :)
        # it is a node
        ch1 = Cache.Forward_Product[M.children[ii, 1]]
        ch2 = Cache.Forward_Product[M.children[ii, 2]]
        bv = Cache.Reverse_Product[ii]
        #         XOp = reshape(DF, size(ch1, 1), size(ch2, 1), size(DF, 2))
        #         @show size(L0), size(bv), size(XOp)
        XOp = reshape(DF, size(ch1, 1), size(ch2, 1), size(DF, 2))
        #         @time @tullio HH[p1, q1, r1, p2, q2, r2] := L0[l1, l2, k] * ch1[p1, 1, k] * ch2[q1, 1, k] * bv[l1, r1, k] * ch1[p2, 1, k] * ch2[q2, 1, k] * bv[l2, r2, k]
        #         @tullio XOp[p2, q2, r2] = (Scaling[k] * ch1[p1, 1, k] * ch2[q1, 1, k] * bv[l, r1, k] * Delta3D[p1, q1, r1]
        #                                     * ch1[p2, 1, k] * ch2[q2, 1, k] * bv[l, r2, k])
        #         println("L0_DF_DF_Delta_parts!")
        @tullio Latent_Delta[l, k] =
            Scaling[k] * bv[l, r1, k] * ch1[p1, 1, k] * ch2[q1, 1, k] * Delta3D[p1, q1, r1]        #=@time=#
        # put this sceond part in cache as it does not change!
        @tullio XOp[p2, q2, r2] =
            (bv[l, r2, k] * Latent_Delta[l, k] * ch1[p2, 1, k] * ch2[q2, 1, k])        #=@time=#
    end
    #     t1 = time()
    #     println("\n -> L0_DF = ", 100*(t1-t0))
    return nothing
end

# same as wDF, except that it only applies to node ii
function L0_DF_parts(
    M::TensorManifold{field},
    X,
    Data::Vararg{AbstractMatrix{T}};
    L0,
    ii::Integer = -1,
    Cache = Make_Cache(M, X, Data...),
) where {field,T}
    #     t0 = time()
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(Data)
            wdim = length(Data)
        else
            wdim = dim
        end
        dd = Data[wdim]
        bv = Cache.Reverse_Product[ii]
        @tullio XO[p, q] := L0[l, k+0] * dd[p, k] * bv[l, q, k]
    else
        # it is a node
        ch1 = Cache.Forward_Product[M.children[ii, 1]]
        ch2 = Cache.Forward_Product[M.children[ii, 2]]
        bv = Cache.Reverse_Product[ii]
        @tullio XOp[p, q, r] := L0[l, k+0] * ch1[p, 1, k] * ch2[q, 1, k] * bv[l, r, k]
        XO = reshape(XOp, :, size(XOp, 3))
    end
    #     t1 = time()
    #     println("\n -> L0_DF = ", 100*(t1-t0))
    return XO
end

function L0_DF(
    M::TensorManifold{field},
    X,
    Data::Vararg{AbstractMatrix{T}};
    L0,
    Cache = Make_Cache(M, X, Data...),
) where {field,T}
    #     @show Tuple(collect(1:length(M.M.manifolds)))
    # @show size(Data[1]), size(L0), size(Cache.Reverse_Product)
    return ArrayPartition(
        map(
            (x) -> L0_DF_parts(M, X, Data..., L0 = L0, ii = x, Cache = Cache),
            Tuple(collect(1:length(M.M.manifolds))),
        ),
    )
end

# L0 is a square matrix ...
# function L0_DF1_DF2_parts(M::TensorManifold{field}, X, L0, dataX, dataY; ii::Integer = -1, DVX = Make_Cache(M, X, dataX), DVY = Make_Cache(M, X, dataY)) where field
#     if is_leaf(M, ii)
#         @tullio XO[p1,q1,p2,q2] := L0[r1,r2,k] * dataX[p1,k] * DVX.Reverse_Product[ii][r1,q1,k] * dataY[p2,k] * DVY.Reverse_Product[ii][r2,q2,k]
#         return XO
#     else
#         # it is a node
# #         @show ii, M.children[ii,1], M.children[ii,2]
#         chX1 = DVX.Forward_Product[M.children[ii,1]]
#         chX2 = DVX.Forward_Product[M.children[ii,2]]
#         bvX = DVX.Reverse_Product[ii]
#         chY1 = DVY.Forward_Product[M.children[ii,1]]
#         chY2 = DVY.Forward_Product[M.children[ii,2]]
#         bvY = DVY.Reverse_Product[ii]
#         @tullio XOp[p1,q1,r1,p2,q2,r2] := L0[l1,l2,k] * chX1[p1,1,k] * chX2[q1,1,k] * bvX[l1,r1,k] * chY1[p2,1,k] * chY2[q2,1,k] * bvY[l2,r2,k]
#         XO = reshape(XOp, size(XOp,1)*size(XOp,2), size(XOp,3), size(XOp,4)*size(XOp,5), size(XOp,6))
#         return XO
#     end
# end

# instead of multiplying the gradient from the left, we are multiplying it from the right
# there is no contraction along the indices of Data...
function DF_dt_parts(
    M::TensorManifold{field},
    X,
    Data...;
    dt,
    ii,
    Cache = Make_Cache(M, X, Data...),
) where {field}
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(Data)
            wdim = length(Data)
        else
            wdim = dim
        end
        l_data = Data[wdim]
        bv = Cache.Reverse_Product[ii]
        # should be made non-allocating
        XO = zeros(eltype(dt), size(bv, 1), size(bv, 3))
        @tullio XO[l, k] = bv[l, q, k] * l_data[p, k+0] * dt[p, q]
    else
        s_l = size(X.x[M.children[ii, 1]], 2)
        s_r = size(X.x[M.children[ii, 2]], 2)
        dtp = reshape(dt, s_l, s_r, :)
        ch1 = Cache.Forward_Product[M.children[ii, 1]]
        ch2 = Cache.Forward_Product[M.children[ii, 2]]
        bv = Cache.Reverse_Product[ii]
        # should be made non-allocating
        XO = zeros(eltype(dtp), size(bv, 1), size(bv, 3))
        @tullio XO[l, k] = ch1[p, 1, k] * ch2[q, 1, k] * bv[l, r, k+0] * dtp[p, q, r]
    end
    #     @show size(Data), size(dt), size(XO)
    return XO
end

function DF_dt(
    M::TensorManifold{field},
    X,
    Data...;
    dt,
    Cache = Make_Cache(M, X, Data...),
) where {field}
    return ArrayPartition(
        map(
            (x, y) -> DF_dt_parts(M, X, Data..., dt = y, ii = x, Cache = Cache),
            Tuple(collect(1:length(M.M.manifolds))),
            dt.x,
        ),
    )
end

# only with respect to the last element of Data
function Jacobian_Add!(
    Jac,
    M::TensorManifold{field},
    X,
    Data...;
    dim,
    Cache = Make_Cache(M, X, Data...),
    Lambda = 1,
) where {field}
    ii = M.dim2ind[dim]
    bv = Cache.Reverse_Product[ii]
    leaf = X.x[ii]
    #     @show dim, size(Jac), size(bv)
    if Lambda == 1
        @tullio Jac[i, q, k] += bv[i, p, k] * leaf[q, p]
    else
        @tullio Jac[i, q, k] += Lambda * bv[i, p, k] * leaf[q, p]
    end
    return nothing
end
