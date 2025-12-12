# SPDX-License-Identifier: EUPL-1.2

function Chebyshev_Grid_Zero(np::Integer)
    return (1 .- cos.(range(0, np - 1) .* pi / (np - 1))) / 2
end

@doc raw"""
    Chebyshev_Grid(np::Integer)

Creates a Chebyshev grid with `np` grid points.
```math
t_j = \cos\frac{(j-1) \pi }{n-1},\;\; j =1,\ldots,n
```
"""
function Chebyshev_Grid(np::Integer)
    return cos.(range(0, np - 1) .* pi / (np - 1))
end

@doc raw"""
    Chebyshev_Grid(np::Integer, a::T, b::T) where {T<:Number}

Creates a Chebyshev grid with `np` grid points within the interval ``[a, b]``.
```math
t_j = \frac{a+b}{2} + \frac{a-b}{2} \cos\frac{(j-1) \pi }{n-1},\;\; j =1,\ldots,n
```
"""
function Chebyshev_Grid(np::Integer, a::T, b::T) where {T <: Number}
    return (a + b) / 2 .+ Chebyshev_Grid(np) .* ((a - b) / 2)
end

function Chebyshev_Mesh(order::Integer, intervals::Integer)
    grid = Chebyshev_Grid_Zero(order)
    mesh = zeros((order - 1) * intervals + 1)
    for k in 1:intervals
        mesh[(1 + (k - 1) * (order - 1)):(1 + k * (order - 1))] .=
            (k - 1) / intervals .+ grid ./ intervals
    end
    return mesh
end

function Barycentric_Interpolation_Weights(t::Vector)
    weights = ones(eltype(t), size(t)...)
    for k in eachindex(weights), j in eachindex(weights)
        if j != k
            weights[j] *= t[j] - t[k]
        end
    end
    weights = 1 ./ weights
    return weights
end

# input (t, val)
# interpolate at (ti)
function Barycentric_Interpolate(t::AbstractVector, val::AbstractArray, ti::AbstractVector)
    val_RS = reshape(val, :, size(val)[end])
    result = zeros(eltype(val), size(val_RS, 1), length(ti))
    weights = Barycentric_Interpolation_Weights(t)
    for k in eachindex(ti)
        denominator = zero(eltype(ti))
        for j in eachindex(t)
            denominator += weights[j] / (ti[k] - t[j])
        end
        for j in eachindex(t)
            numerator = weights[j] / (ti[k] - t[j])
            if ti[k] == t[j]
                for l in axes(result, 1)
                    result[l, k] += val_RS[l, j]
                end
            else
                for l in axes(result, 1)
                    result[l, k] += (numerator / denominator) * val_RS[l, j]
                end
            end
        end
    end
    return result
end

# function represented by
# at points 't[.]'
# interpolate at points ti[.]
@doc raw"""
    Barycentric_Interpolation_Matrix(t::AbstractVector, ti::AbstractVector)

Creates a matrix that performs a polynomial interpolation from grid `t` to grid `ti`.
"""
function Barycentric_Interpolation_Matrix(t::AbstractVector, ti::AbstractVector)
    weights = Barycentric_Interpolation_Weights(t)
    interp_matrix = zeros(eltype(ti), length(ti), length(t))
    for k in eachindex(ti)
        denominator = zero(eltype(ti))
        for j in eachindex(t)
            denominator += weights[j] / (ti[k] - t[j])
        end
        for j in eachindex(t)
            numerator = weights[j] / (ti[k] - t[j])
            if ti[k] == t[j]
                interp_matrix[k, j] = 1.0
            else
                interp_matrix[k, j] = numerator / denominator
            end
        end
    end
    return interp_matrix
end

@doc raw"""
    Barycentric_Interpolation_Matrix(order::Integer, mesh::Vector, ti::AbstractVector)

Creates a matrix that performs a polynomial interpolation from a `mesh`, where each interval has `order` number of Chebyshev points.
The target points are in `grid`.
"""
function Barycentric_Interpolation_Matrix(order::Integer, mesh::Vector, ti::AbstractVector)
    interp_matrix = zeros(eltype(ti), length(ti), length(mesh))
    sparse_mesh = view(mesh, 1:(order - 1):length(mesh))
    for k in 2:length(sparse_mesh)
        mesh_range = (1 + (k - 2) * (order - 1)):(1 + (k - 1) * (order - 1))
        idx = ifelse(
            k == 2,
            findall(x -> ((x >= sparse_mesh[k - 1]) && (x <= sparse_mesh[k])), ti),
            findall(x -> ((x > sparse_mesh[k - 1]) && (x <= sparse_mesh[k])), ti),
        )
        interp_matrix[idx, mesh_range] .=
            Barycentric_Interpolation_Matrix(mesh[mesh_range], ti[idx])
    end
    return interp_matrix
end

function Barycentric_Differentiation_Matrix(t::AbstractVector)
    weights = Barycentric_Interpolation_Weights(t)
    np = length(t)
    D = zeros(eltype(t), np, np)
    for i in Base.OneTo(np)
        Dsum = zero(eltype(t))
        for j in Base.OneTo(i - 1)
            temp = (weights[j] / weights[i]) / (t[i] - t[j])
            D[i, j] = temp
            Dsum += temp
        end
        for j in (i + 1):np
            temp = (weights[j] / weights[i]) / (t[i] - t[j])
            D[i, j] = temp
            Dsum += temp
        end
        D[i, i] = -Dsum
    end
    return D
end

# Evaluates the Chebyshev Polynomials up to order - 1
# on the grid 't' scale to the interval [a, b]
# order is order - 1
function Chebyshev_Evalation_Matrix!(
        result,
        t::AbstractVector,
        order::Integer,
        a::Number,
        b::Number,
    )
    ti = (t .- (a + b) / 2) ./ ((b - a) / 2)
    for k in 1:order
        if k == 1
            result[:, k] .= 1
        elseif k == 2
            result[:, k] .= ti
        else
            result[:, k] .= 2 * ti .* result[:, k - 1] - result[:, k - 2]
        end
    end
    return result
end

function Chebyshev_Evalation_Matrix(t::AbstractVector, order::Integer, a::Number, b::Number)
    result = zeros(eltype(t), length(t), order)
    return Chebyshev_Evalation_Matrix!(
        result,
        t::AbstractVector,
        order::Integer,
        a::Number,
        b::Number,
    )
end

function Chebyshev_Derivative(order)
    D = zeros(order, order)
    for k in 1:order
        if k == 1
            D[k, range(k + 1, order, step = 2)] .= range(k, order - 1, step = 2)
        else
            D[k, range(k + 1, order, step = 2)] .= range(2 * k, 2 * order - 1, step = 4)
        end
    end
    return D
end

function Clenshaw_Curtis_Matrix(order)
    np = order - 1
    result = [(-1)^k * 2 * cos(pi * j * k / np) / np for k in 0:np, j in 0:np]
    result[:, 1] .*= 0.5
    result[:, end] .*= 0.5
    result[1, :] .*= 0.5
    result[end, :] .*= 0.5
    return result
end

function Clenshaw_Interpolate(t::AbstractVector, order::Integer, mesh::Vector)
    ITP = zeros(eltype(t), length(t), length(mesh))
    DTP = zeros(eltype(t), length(t), length(mesh))
    sparse_mesh = view(mesh, 1:(order - 1):length(mesh))
    DD = Chebyshev_Derivative(order)
    for k in 2:length(sparse_mesh)
        mesh_range = (1 + (k - 2) * (order - 1)):(1 + (k - 1) * (order - 1))
        start = ifelse(
            k == 2,
            findfirst(x -> (x >= sparse_mesh[k - 1]), t),
            findfirst(x -> (x > sparse_mesh[k - 1]), t),
        )
        fin = findlast(x -> (x <= sparse_mesh[k]), t)
        if isnothing(start) || isnothing(fin)
            #             print("Interval ", k, " is non-existent: ", start, " ", fin)
            continue
        end
        MM_TT = Chebyshev_Evalation_Matrix(
            t[start:fin],
            order,
            sparse_mesh[k - 1],
            sparse_mesh[k],
        )
        I_MM_ID = Clenshaw_Curtis_Matrix(order)
        ITP[start:fin, mesh_range] .= MM_TT * I_MM_ID
        DTP[start:fin, mesh_range] .=
            (2 / (sparse_mesh[k] - sparse_mesh[k - 1])) * MM_TT * DD * I_MM_ID
    end
    return ITP, DTP
end
