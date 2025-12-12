# SPDX-License-Identifier: EUPL-1.2

function Givens_Rotation(a, b)
    r = sqrt(a^2 + b^2)
    c = a / r
    s = -b / r
    return c, s
end

function Hessenberg_Shift_By_Givens!(Q, H, A, SH; Align = false)
    H .= A
    Q .= 0
    for p in axes(Q, 1), q in axes(Q, 2)
        Q[p, q, p] = 1
    end
    c = zeros(eltype(A), size(A, 2))
    s = zeros(eltype(A), size(A, 2))
    for i in 2:(size(A, 1) - 1)
        for j in (i + 1):size(A, 1)
            # calculate rotation to all parts
            for q in axes(H, 2)
                if (H[i, q, i - 1]^2 + H[j, q, i - 1]^2) < 16 * eps(1.0)
                    print(i, ".", j, "|")
                    c[q] = 1
                    s[q] = 0
                    continue
                end
                c[q], s[q] = Givens_Rotation(H[i, q, i - 1], H[j, q, i - 1])
                if (q > 1) && Align
                    H_i_q_i_m_1 = H[i, q, i - 1] * c[q] - H[j, q, i - 1] * s[q]
                    tmp_j = H[i, q, i - 1] * s[q] + H[j, q, i - 1] * c[q]
                    if abs(H[i, q - 1, i - 1] - H_i_q_i_m_1) > abs(H[i, q - 1, i - 1] + H_i_q_i_m_1)
                        c[q] = -c[q]
                        s[q] = -s[q]
                        print("/")
                    end
                end
                # from the left
                for p in axes(H, 3)
                    tmp_i = H[i, q, p] * c[q] - H[j, q, p] * s[q]
                    tmp_j = H[i, q, p] * s[q] + H[j, q, p] * c[q]
                    H[i, q, p] = tmp_i
                    H[j, q, p] = tmp_j
                end
                H[j, q, i - 1] = 0
                for p in axes(Q, 1)
                    tmp_i = Q[p, q, i] * c[q] - Q[p, q, j] * s[q]
                    tmp_j = Q[p, q, i] * s[q] + Q[p, q, j] * c[q]
                    Q[p, q, i] = tmp_i
                    Q[p, q, j] = tmp_j
                end
            end
            # shift the rotations
            @tullio cs[k] := c[l] * SH[k, l]
            @tullio ss[k] := s[l] * SH[k, l]
            for q in axes(H, 2)
                scale = 1 / sqrt(cs[q]^2 + ss[q]^2)
                for p in axes(H, 1)
                    tmp_i = H[p, q, i] * cs[q] - H[p, q, j] * ss[q]
                    tmp_j = H[p, q, i] * ss[q] + H[p, q, j] * cs[q]
                    H[p, q, i] = scale * tmp_i
                    H[p, q, j] = scale * tmp_j
                end
            end
        end
    end
    return nothing
end

@inline function QR_By_Givens!(Q, Q_SH, R, A, s, c, s_SH, c_SH, SH; Align = false)
    R .= A
    Q .= 0
    Q_SH .= 0
    for p in axes(Q, 1), q in axes(Q, 2)
        Q[p, q, p] = 1
        Q_SH[p, q, p] = 1
    end
    for i in 1:(size(A, 3) - 1)
        j = i + 1
        for k in axes(A, 2)
            if (R[i, k, i]^2 + R[j, k, i]^2) < 16 * eps(1.0)
                c[k] = 1
                s[k] = 0
                continue
            end
            c[k], s[k] = Givens_Rotation(R[i, k, i], R[j, k, i])
            if (k > 1) && Align
                R_i_k_i = R[i, k, i] * c[k] - R[j, k, i] * s[k]
                if abs(R[i, k - 1, i] - R_i_k_i) > abs(R[i, k - 1, i] + R_i_k_i)
                    c[k] = -c[k]
                    s[k] = -s[k]
                    print("+")
                end
            end
            #
            for p in axes(R, 3)
                tmp_i = R[i, k, p] * c[k] - R[j, k, p] * s[k]
                tmp_j = R[i, k, p] * s[k] + R[j, k, p] * c[k]
                R[i, k, p] = tmp_i
                R[j, k, p] = tmp_j
            end
            R[j, k, i] = 0
            for p in axes(Q, 1)
                tmp_i = Q[p, k, i] * c[k] - Q[p, k, j] * s[k]
                tmp_j = Q[p, k, i] * s[k] + Q[p, k, j] * c[k]
                Q[p, k, i] = tmp_i
                Q[p, k, j] = tmp_j
            end
        end
        @tullio c_SH[k] = c[q] * SH[k, q]
        @tullio s_SH[k] = s[q] * SH[k, q]
        for k in axes(A, 2)
            scale = 1 / sqrt(c_SH[k]^2 + s_SH[k]^2)
            for p in axes(Q_SH, 1)
                tmp_i = Q_SH[p, k, i] * c_SH[k] - Q_SH[p, k, j] * s_SH[k]
                tmp_j = Q_SH[p, k, i] * s_SH[k] + Q_SH[p, k, j] * c_SH[k]
                Q_SH[p, k, i] = scale * tmp_i
                Q_SH[p, k, j] = scale * tmp_j
            end
        end
    end
    return nothing
end

function QR_Shift_Iteration(AA, SH; maxiter = 12000, Align = false)
    BB = deepcopy(AA)
    QQ = zero(BB)
    RR = zero(BB)
    QQ_A = zero(BB)
    QQ_SH = zero(BB)
    QQ_A_SH = zero(BB)
    cc = zeros(eltype(BB), size(BB, 2))
    ss = zero(cc)
    cc_SH = zero(cc)
    ss_SH = zero(cc)
    for p in axes(QQ_A, 1), q in axes(QQ_A, 2)
        QQ_A[p, q, p] = 1
        QQ_A_SH[p, q, p] = 1
    end
    for it in 1:maxiter
        #         QR_By_Givens!(QQ, RR, BB; Align = Align)
        QR_By_Givens!(QQ, QQ_SH, RR, BB, ss, cc, ss_SH, cc_SH, SH; Align = Align)
        #         @tullio QQ_SH[s, r, j] = QQ[s, q, j] * SH[r, q]
        @tullio BB[i, r, j] = RR[i, r, s] * QQ_SH[s, r, j]
        for p in axes(QQ, 2)
            RR_R = view(RR, :, p, :)
            QQ_R = view(QQ, :, p, :)
            QQ_A_R = view(QQ_A, :, p, :)
            mul!(RR_R, QQ_A_R, QQ_R)
            QQ_A_R .= RR_R
            #
            QQ_SH_R = view(QQ_SH, :, p, :)
            QQ_A_SH_R = view(QQ_A_SH, :, p, :)
            mul!(RR_R, QQ_A_SH_R, QQ_SH_R)
            QQ_A_SH_R .= RR_R
        end
        Max_Value = maximum(abs, BB)
        #         Max_Error = zero(eltype(BB))
        #         Sub_Diagonals = 0
        #         for p = 1:size(BB, 1)-1, q in axes(BB, 2)
        #             tr = BB[p, q, p] + BB[p+1, q, p+1]
        #             det = BB[p, q, p] * BB[p+1, q, p+1] - BB[p+1, q, p] * BB[p, q, p+1]
        #             if tr^2 - 4 * det > 0
        #                 Sub_Diagonals += 1
        #                 if Max_Error < abs(BB[p+1, q, p])
        #                     Max_Error = abs(BB[p+1, q, p])
        #                 end
        #             end
        #         end
        ee = [sum(BB[p + 1, :, p] .^ 2) for p in 1:(size(BB, 3) - 1)]
        Index = 0
        Prev_Index = 1
        Max_Dim = 0
        for k in eachindex(ee)
            if ee[k] < 2^10 * eps(Max_Value)
                Prev_Index = Index
                Index = k
                if Max_Dim < Index - Prev_Index
                    Max_Dim = Index - Prev_Index
                end
            end
        end
        if Max_Dim < size(BB, 3) - Index
            Max_Dim = size(BB, 3) - Index
        end
        print("|", Max_Dim)
        if Max_Dim <= 2
            println("Converged in ", it, " iterations.")
            @show ee
            @tullio E[p, j, q] := QQ_A[p, j, i] * QQ_A[q, j, i]
            for p in axes(E, 1), q in axes(E, 2)
                E[p, q, p] -= 1
            end
            println("Error in forward transformation: ", norm(E))
            @tullio E[p, j, q] = QQ_A_SH[p, j, i] * QQ_A_SH[q, j, i]
            for p in axes(E, 1), q in axes(E, 2)
                E[p, q, p] -= 1
            end
            println("Error in shift transformation: ", norm(E))
            return BB, QQ_A, QQ_A_SH
        end
    end
    println("Failed to converge in ", maxiter, " iterations.")
    @tullio E[p, j, q] := QQ_A[p, j, i] * QQ_A[q, j, i]
    for p in axes(E, 1), q in axes(E, 2)
        E[p, q, p] -= 1
    end
    println("Error in forward transformation: ", norm(E))
    @tullio E[p, j, q] = QQ_A_SH[p, j, i] * QQ_A_SH[q, j, i]
    for p in axes(E, 1), q in axes(E, 2)
        E[p, q, p] -= 1
    end
    println("Error in shift transformation: ", norm(E))
    return BB, QQ_A, QQ_A_SH
end

function Schur_Shift(BB, SH; Align = false, maxiter = 12000)
    QH = zero(BB)
    HH = zero(BB)
    #     println("Original matrix")
    #     for k in axes(BB, 2)
    #         display(BB[:, k, :])
    #     end
    Hessenberg_Shift_By_Givens!(QH, HH, BB, SH; Align = Align)
    #
    #     println("Hessenberg matrix")
    #     for k in axes(HH, 2)
    #         display(HH[:, k, :])
    #     end
    #     println("Hessenberg transformation matrix")
    #     for k in axes(QH, 2)
    #         display(QH[:, k, :])
    #     end
    #
    CC, QQ, QQ_SH = QR_Shift_Iteration(HH, SH; Align = Align, maxiter = maxiter)
    @tullio QQ_A[i, j, k] := QH[i, j, p] * QQ[p, j, k]
    @tullio QQ_A_SH[i, j, k] := QQ_A[i, p, k] * SH[j, p]
    # testing
    @tullio BB2[i, p, l] := QQ_A[i, p, j] * CC[j, p, k] * QQ_A_SH[l, p, k]
    @show norm(BB2 - BB)
    return CC, QQ_A, QQ_A_SH, HH
end

function Solve_For_Vector_Bundle(A2, A3, B23, SH)
    M = zeros(eltype(A2), size(B23)..., size(B23)...)
    for p in axes(B23, 1), q in axes(B23, 2), r in axes(B23, 3)
        for i in axes(B23, 1), j in axes(B23, 2), k in axes(B23, 3)
            M[i, j, k, p, q, r] =
                A2[i, j, p] * I[j, q] * I[k, r] - I[i, p] * SH[q, j] * A3[r, j, k]
        end
    end
    M_R = reshape(M, length(B23), length(B23))
    U = M_R \ vec(B23)
    return reshape(U, size(B23)...)
end

function Schur_To_Diagonal(CC, SH)
    CC_Avg = dropdims(sum(abs.(CC), dims = 2) ./ size(CC, 2), dims = 2)
    Max_Value = maximum(CC_Avg)
    Bundles = []
    let Index = 1
        for k in 1:(size(CC_Avg, 1) - 1)
            if CC_Avg[k + 1, k] < 2^12 * eps(Max_Value)
                push!(Bundles, Index:k)
                Index = k + 1
            end
        end
        push!(Bundles, Index:size(CC_Avg, 1))
    end

    UU_A = zero(CC)
    for k in 1:(length(Bundles) - 1)
        UU = Solve_For_Vector_Bundle(
            CC[Bundles[k], :, Bundles[k]],
            CC[(Bundles[k][end] + 1):end, :, (Bundles[k][end] + 1):end],
            CC[Bundles[k], :, (Bundles[k][end] + 1):end],
            SH,
        )
        UU_A[Bundles[k], :, (Bundles[k][end] + 1):end] .= UU
    end
    for k in eachindex(Bundles)
        for q in axes(UU_A, 2)
            UU_A[Bundles[k], q, Bundles[k]] .= I[Bundles[k], Bundles[k]]
        end
    end

    @tullio UU_A_SH[i, j, k] := UU_A[i, p, k] * SH[p, j]
    UU_A_INV = zero(UU_A)
    for p in axes(UU_A, 2)
        UU_A_INV[:, p, :] = inv(UU_A[:, p, :])
    end
    UU_A_SH_INV = zero(UU_A)
    for p in axes(UU_A, 2)
        UU_A_SH_INV[:, p, :] = inv(UU_A_SH[:, p, :])
    end
    @tullio DD[i, p, l] := UU_A_SH[i, p, j] * CC[j, p, k] * UU_A_INV[k, p, l]
    return DD, UU_A_SH, UU_A_INV, UU_A, UU_A_SH_INV, Bundles
end

function Transfer_Operator_Transposed(BB, SH)
    TR = zeros(eltype(BB), size(BB, 3), size(SH, 2), size(BB, 1), size(SH, 1))
    @tullio TR[i, j, p, q] = SH[q, j] * BB[p, j, i]
    return reshape(TR, size(BB, 3) * size(SH, 2), size(BB, 1) * size(SH, 1))
end

function Total_Variation_Of_Bundles(Skew_Dimension, vectors)
    r1 = reshape(vectors, :, Skew_Dimension, size(vectors, 2))
    df1 = r1 - cat(r1[:, 2:end, :], r1[:, [1], :], dims = 2)
    tv = vec(sum(sqrt.(real.(sum(df1 .* conj.(df1), dims = 1))), dims = 2))
    return tv
end

# calculate index of smallest total variation among vectors
function Minimum_Total_Variation_Of_Bundles(Skew_Dimension, vectors)
    tv = Total_Variation_Of_Bundles(Skew_Dimension, vectors)
    return argmin(tv)
end

function Reduce_Bundles(AA, Bundles, SH)
    Skew_Dimension = size(SH, 1)
    V_R = zero(AA)
    Lambda = zero(AA)
    for Bundle_Index in Bundles
        BB = view(AA, Bundle_Index, :, Bundle_Index)
        TR = Transfer_Operator_Transposed(BB, SH)
        values, vectors = eigen(TR)
        id = Minimum_Total_Variation_Of_Bundles(Skew_Dimension, vectors)
        if size(BB, 1) == 1
            if !isreal.(values[id])
                println(
                    "Reduce_Bundles: Complex eigenvalue of a one dimensional vector bundle.",
                )
            end
            V_R_R = view(V_R, Bundle_Index, :, Bundle_Index)
            Scale3 = copysign(
                sqrt(real(dot(vectors[:, id], vectors[:, id]))),
                real(mean(vectors[:, id])),
            )
            #             @show Scale3 / Skew_Dimension
            V_R_R .=
                real.(reshape(vectors[:, id], 1, :, 1)) .* (sqrt(Skew_Dimension) / Scale3)
            Lambda[Bundle_Index, :, Bundle_Index] .= real(values[id])
        elseif size(BB, 1) == 2
            V_R_R = view(V_R, Bundle_Index, :, Bundle_Index)
            VV = cat(
                real.(reshape(vectors[:, id], 1, 2, :)),
                imag.(reshape(vectors[:, id], 1, 2, :)),
                dims = 1,
            )
            Sign = copysign(one(eltype(VV)), mean(VV))
            @tullio Correlation[i, j] := VV[i, k, p] * VV[j, k, p]
            F = svd(Correlation)
            Scale = Diagonal(sqrt(Skew_Dimension) ./ sqrt.(F.S)) * F.U'
            @tullio V_R_R[i, j, k] = Scale[i, p] * VV[p, k, j] * Sign
            @tullio Scale3[i, j] := V_R_R[i, k, p] * V_R_R[j, k, p]
            #             @show Scale3 / (Skew_Dimension ^ 2)
            #             Lambda_Constant = inv(Scale) * SMatrix{2,2}(real(values[id]), -imag(values[id]), imag(values[id]), real(values[id])) * Scale
            Lambda_Constant =
                Scale *
                SMatrix{2, 2}(
                real(values[id]),
                imag(values[id]),
                -imag(values[id]),
                real(values[id]),
            ) *
                inv(Scale)
            #             Lambda_Constant = SMatrix{2,2}(real(values[id]), -imag(values[id]), imag(values[id]), real(values[id]))
            Lambda[Bundle_Index, :, Bundle_Index] .= reshape(Lambda_Constant, 2, 1, 2)
        else
            println(
                "Reduce_Bundles: sub-bundle dimension ",
                size(BB, 1),
                " is not supported.",
            )
            @show Bundles
            display(BB[:, 1, :])
            display(SH)
            return nothing
        end
    end
    # rearrange bundles by magnitude
    Magnitudes = [
        maximum(abs.(eigvals(Lambda[Bundle_Index, 1, Bundle_Index]))) for
            Bundle_Index in Bundles
    ]
    if !all(Magnitudes[2:end] .<= Magnitudes[1:(end - 1)])
        println("Reduce_Bundles: eigenvalues are not sorted.")
    end
    W_R = zero(V_R)
    W_R_SH = zero(V_R)
    # V_R = zero(V_R)
    V_R_SH = zero(V_R)
    for Bundle_Index in Bundles
        W_R_R = view(W_R, Bundle_Index, :, Bundle_Index)
        W_R_SH_R = view(W_R_SH, Bundle_Index, :, Bundle_Index)
        V_R_R = view(V_R, Bundle_Index, :, Bundle_Index)
        V_R_SH_R = view(V_R_SH, Bundle_Index, :, Bundle_Index)
        @tullio V_R_SH_R[i, k, j] = V_R_R[i, l, j] * SH[l, k]
        for p in axes(V_R_SH_R, 2)
            W_R_R[:, p, :] .= inv(V_R_R[:, p, :])
            W_R_SH_R[:, p, :] .= inv(V_R_SH_R[:, p, :])
        end
    end
    @tullio KK[i, k, r] := W_R_SH[i, k, j] * Lambda[j, k, l] * V_R[l, k, r]
    @show norm(AA - KK)
    @tullio Lambda_P[i, k, r] := V_R_SH[i, k, j] * AA[j, k, l] * W_R[l, k, r]
    @show norm(Lambda - Lambda_P)
    #     display(Lambda[:, 10, :])
    #     display(Lambda_P[:, 10, :])
    #     display(Lambda[:, 10, :] - Lambda_P[:, 10, :])
    return Lambda, W_R, V_R_SH, V_R, W_R_SH
end

function Bundle_Decomposition(BB, SH; Align = true, maxiter = 12000)
    CC, QQ_A, QQ_A_SH, HH = Schur_Shift(BB, SH; Align = Align, maxiter = maxiter)
    #     println("Schur matrix")
    #     for k in axes(CC, 2)
    #         display(CC[:, k, :])
    #     end
    Unreduced_Model, UU_A_SH, UU_A_INV, UU_A, UU_A_SH_INV, Bundles =
        Schur_To_Diagonal(CC, SH)
    #
    @tullio QQ_F[i, j, k] := QQ_A[i, j, p] * UU_A_SH_INV[p, j, k]
    @tullio Data_Encoder[i, j, k] := UU_A[i, j, p] * QQ_A_SH[k, j, p]
    @tullio BB3[i, p, l] := QQ_F[i, p, j] * Unreduced_Model[j, p, k] * Data_Encoder[k, p, l]
    @show norm(BB3 - BB)
    #
    @tullio QQ_R[i, j, k] := UU_A_SH[i, j, p] * QQ_A[k, j, p]
    @tullio Data_Decoder[i, j, k] := QQ_A_SH[i, j, p] * UU_A_INV[p, j, k]
    @tullio DD2[i, p, l] := QQ_R[i, p, j] * BB[j, p, k] * Data_Decoder[k, p, l]
    @show norm(DD2 - Unreduced_Model)
    #
    return Unreduced_Model, Data_Encoder, Data_Decoder, Bundles
end
