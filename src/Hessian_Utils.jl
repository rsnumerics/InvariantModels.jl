# SPDX-License-Identifier: EUPL-1.2

function riemannian_Hessian!(M::Euclidean, Y, p, G, H, X)
    copyto!(Y, H)
    return Y
end

function riemannian_Hessian!(M::Stiefel, Y, p, G, H, X)
    return ManifoldDiff.riemannian_Hessian!(M, Y, p, G, H, X)
end

function ManifoldsBase.Weingarten!(::AbstractSphere, Y, p, X, V)
    Y .= -real(dot(p, V)) .* X
    return Y
end
