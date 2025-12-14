# SPDX-License-Identifier: EUPL-1.2

# the interpolation function
function psi(theta, np)
    th = mod(theta + pi, 2 * pi) - pi
    bound = (eps(1.0) * 5760 / (7 + 3 * np^4))^0.25
    if abs(th) < bound
        return 1 + ((1 - np^2) * th^2) / 24
    else
        return sin((np / 2) * th) / (np * sin(th / 2))
    end
end

function D_psi(theta, np)
    th = mod(theta + pi, 2 * pi) - pi
    bound = (eps(1.0) * 5760 / (7 + 3 * np^4))^0.25
    if abs(th) < bound
        return ((-1 + np^2) * th * (-120 + (-7 + 3 * np^2) * th^2)) / 1440
    else
        return (
            csc(th / 2) * (np * cos((np * th) / 2) - cot(th / 2) * sin((np * th) / 2.0))
        ) / (2 * np)
    end
end

# creates a vector that interpolates ths scalar valued 'theta' to the uniform grid
# the grid must be created by getgrid(n)
"""
    Fourier_Interpolate(grid, theta::Number)

Evaluates the set of Dirichlet kernels specific to the uniform `grid` at point theta.
The result is a vector.
"""
function Fourier_Interpolate(grid, theta::Number)
    psi2 = (x) -> psi(x, length(grid))
    return psi2.(theta .- grid)
end

# """
#     Rigid_Rotation!(Alpha_t, Grid, Omega_ODE::Number, IC_Alpha, t::Number)

# TBW
# """
# function Rigid_Rotation!(Alpha_t, Grid, Omega_ODE::Number, IC_Alpha, t::Number)
#     for j in eachindex(Grid)
#         Alpha_t[j] = 0
#         for k in eachindex(Grid)
#             Alpha_t[j] += psi(Grid[j] - Grid[k] - Omega_ODE * t, length(Grid)) * IC_Alpha[k]
#         end
#     end
#     return Alpha_t
# end

# function Rigid_Rotation!(
#     Alpha_t,
#     Grid,
#     Omega_ODE::Number,
#     IC_Alpha,
#     t::AbstractArray{T,1},
# ) where {T}
#     for k in eachindex(t)
#         Rigid_Rotation!(view(Alpha_t, :, k), Grid, Omega_ODE::Number, IC_Alpha, t[k])
#     end
#     return Alpha_t
# end

"""
    Rigid_Rotation_Matrix!(Alpha_Matrix, Grid, Omega_ODE::Number, t::Number)

Creates the fundamental solution matrix for the rigid rotation on the circle in the space of the uniform `Grid` on the circle.
The angular speed of the rotation is `Omega_ODE` and the result is written into `Alpha_Matrix`. The indepedent variabe is `t`.
"""
function Rigid_Rotation_Matrix!(Alpha_Matrix, Grid, Omega_ODE::Number, t::Number)
    for j in eachindex(Grid)
        for k in eachindex(Grid)
            Alpha_Matrix[j, k] = psi(Grid[j] - Grid[k] - Omega_ODE * t, length(Grid))
        end
    end
    return Alpha_Matrix
end

"""
    Rigid_Rotation_Generator(Grid, Omega_ODE::Number)

This is the infinitesimal generator of the rigid rotation with `Omega_ODE` angular speed.
"""
function Rigid_Rotation_Generator(Grid, Omega_ODE::Number)
    Generator = zeros(eltype(Grid), length(Grid), length(Grid))
    for j in eachindex(Grid)
        for k in eachindex(Grid)
            Generator[j, k] = -Omega_ODE * D_psi(Grid[j] - Grid[k], length(Grid))
        end
    end
    return Generator
end

"""
    Fourier_Interpolate(grid, theta::AbstractArray{T,1}) where {T}

Evaluates the set of Dirichlet kernels specific to the uniform `grid` at all points in `theta`.
The result is a matrix, where each column corresponds to a value in `theta`.
"""
function Fourier_Interpolate(grid, theta::AbstractArray{T, 1}) where {T}
    W = zeros(T, length(grid), length(theta))
    psi2 = (x) -> psi(x, length(grid))
    for k in eachindex(theta)
        W[:, k] .= psi2.(theta[k] .- grid)
    end
    return W
end

function diffpsi(k::Integer, np)
    return iszero(k) ? zero(sin(k)) : (-1)^k / sin(k * pi / np) / 2
end

function differentialOperator(grid)
    DD = zeros(Float64, length(grid), length(grid))
    for j in eachindex(grid), k in eachindex(grid)
        DD[j, k] = diffpsi(j - k, length(grid))
    end
    return DD
end
#

"""
    Fourier_Grid(Phase_Dimension::Integer)

Creates a uniform grid on the ``[0, 2\\pi)`` interval.
"""
function Fourier_Grid(Phase_Dimension::Integer)
    return range(
        0,
        2 * pi * (Phase_Dimension - 1) / Phase_Dimension,
        length = Phase_Dimension,
    )
end

function Shift_Operator(grid, omega)
    SH = zeros(typeof(omega), length(grid), length(grid))
    for j in eachindex(grid), k in eachindex(grid)
        SH[j, k] = psi(grid[j] - grid[k] - omega, length(grid))
    end
    return SH
end

function Shift_Operator(grid, theta, omega)
    return Fourier_Interpolate(grid .- omega, theta)
end

function Fourier_Differential(Phase_Dimension::Integer)
    DD = zeros(Float64, Phase_Dimension, Phase_Dimension)
    for j in axes(DD, 1), k in axes(DD, 2)
        DD[j, k] = diffpsi(j - k, Phase_Dimension)
    end
    return DD
end

#     Fourier_Grid_Of_Order(n::Integer)
# Returns a uniform grid on the interval ``[0, 2 \pi)`` with ``2n + 1`` number of points. The end point ``2\pi`` is not part of the grid.
# The number ``n`` corresponds to the number of Fourier harmonics that can be represented on the grid.
function Fourier_Grid_Of_Order(n::Integer)
    return range(0.0, 4 * n * pi / (2 * n + 1), length = 2 * n + 1)
end

function FourierMatrix(grid)
    fourier_order = div(length(grid) - 1, 2)
    @assert length(grid) == 2 * fourier_order + 1 "Incorrect grid size"
    tr =
        [exp(-k * 1im * grid[l]) for k in (-fourier_order):fourier_order, l in 1:length(grid)]
    return tr
end
