function Total_Variation_Of_Vectors(Skew_Dimension, vectors)
    r1 = reshape(vectors, Skew_Dimension, :, size(vectors, 2))
    df1 = r1 - cat(r1[2:end, :, :], r1[[1], :, :], dims=1)
    tv = vec(sum(sqrt.(real.(sum(df1 .* conj.(df1), dims=2))), dims=1))
    return tv
end

# calculate index of smallest total variation among vectors
function Minimum_Total_Variation_Of_Vectors(Skew_Dimension, vectors)
    tv = Total_Variation_Of_Vectors(Skew_Dimension, vectors)
    return argmin(tv)
end

function Eigenvector_Adjacency(Skew_Dimension, vectors)
    r1 = reshape(vectors, Skew_Dimension, :, size(vectors, 2))
    adj = zeros(size(vectors, 2), size(vectors, 2))
    for k in axes(r1, 3), j in axes(r1, 3)
        for p in axes(r1, 1)
            v_j = r1[p, :, j]
            v_k = r1[p, :, k]
            a = sum(v_k .* conj.(v_j)) / sum(v_j .* conj.(v_j))
            adj[j, k] += norm(a * v_j - v_k) / sqrt(abs(a))
        end
        adj[j, k] /= size(r1, 1)
        if norm(conj.(vectors[:, j]) - vectors[:, k]) < 32 * eps(1.0) && !all(isreal(vectors[:, j]))
            adj[j, k] += 1.0
        end
    end
    return adj
end

# internal function
# finds a complex cluster that has no complex conjugate pairs
# because conjugate pairs should not belong to the same cluster
function Find_NonConjugate_Cluster(Skew_Dimension, values, adj, remset, id)
    clord = remset[sortperm(adj[id, remset])]
    cl = [clord[1]]
    len = 1
    while (length(cl) < Skew_Dimension) && (len < length(clord))
        cj = conj.(values[cl])
        if !in(conj(values[clord[len+1]]), values[cl])
            push!(cl, clord[len+1])
        end
        len = len + 1
    end
    return cl
end

# finds clusters of eigenvectors and the one with smallest total variation in each cluster
# neds to take into account complex conjugate pairs, as they should be in the same cluster
function Find_Eigenvector_Clusters(Skew_Dimension, values, vectors)
    adj = Eigenvector_Adjacency(Skew_Dimension, vectors)
    tv = Total_Variation_Of_Vectors(Skew_Dimension, vectors)
    adj = (adj + transpose(adj))

    # creating possible clusters
    remset = collect(1:size(vectors, 2))
    clusters = []
    clrepr = []
    clgood = []
    while length(remset) >= Skew_Dimension
        id = remset[argmin(tv[remset])]
        if isreal(values[id]) || (length(remset) < 2 * Skew_Dimension)
            print("   - real cluster => ", isreal(values[id]), " ")
            cl = remset[sortperm(adj[id, remset])[1:Skew_Dimension]]
            push!(clusters, cl)
            setdiff!(remset, cl)
            clr = findall(isreal.(values[cl]))
            if isreal(values[id]) || isempty(clr)
                println(length(cl), " ", id, " ", log(values[id]) / 3.7)
                push!(clrepr, id)
            else
                id_r = cl[clr[argmin(tv[cl[clr]])]]
                println(length(cl), " ", id_r, " ", log(values[id_r]) / 3.7)
                push!(clrepr, id_r)
            end
        else
            print("   - complex cluster ")
            # find a cluster that has no complex conjugate pairs
            cl = Find_NonConjugate_Cluster(Skew_Dimension, values, adj, remset, id)
            print(length(cl), " ", id, " ", log(values[id]) / 3.7, " | ")
            push!(clusters, cl)
            setdiff!(remset, cl)
            push!(clrepr, id)
            # finding the pair
            cc = findfirst(isapprox(conj(values[id])), values[remset])
            if !isnothing(cc)
                id_c = remset[cc]
                # a) just find the complex conjugate pairs
                #                     cl_c_id = [findfirst(isapprox(a), values[remset]) for a in conj.(values[cl])]
                #                     cl_c = remset[cl_c_id[findall(.! isnothing.(cl_c_id))]]
                # b) find the cluster from adjacency
                cl_c = Find_NonConjugate_Cluster(Skew_Dimension, values, adj, remset, id_c)
                println(length(cl_c), " ", id_c, " ", log(values[id_c]) / 3.7)
                push!(clusters, cl_c)
                setdiff!(remset, cl_c)
                push!(clrepr, id_c)
            else
                println("MISSING")
            end
        end
    end
    #         @show [length(cl) for cl in clusters]
    #         @show clrepr
    #         @show sort(vcat(clusters...))
    if !isempty(remset)
        println("Find_Eigenvector_Clusters: Remaining eigenvalues")
        display(remset)
    end
    return clusters, clrepr, tv
end

function Decompose_Eigenvectors(AA_Right, Skew_Dimension; Time_Step, sparse=false, dims, ODE=false)
    ############ CALCULATING LEFT AND RIGHT EIGENVECTORS ############
    if ODE
        importance = x -> real(x)
        freq_measure = x -> abs(imag(x))
        to_ode = x -> x
    else
        importance = x -> abs(x)
        freq_measure = x -> abs(angle(x))
        to_ode = x -> log(x) / Time_Step
    end
    if sparse
        decomp, history = partialschur(AA_Right, nev=min(3 * dims * Skew_Dimension, size(AA_Right, 1)), which=:LM)
        ll, vv = partialeigen(decomp)
        decomp_T, history_T = partialschur(
            AA_Right, nev=min(3 * dims * Skew_Dimension, size(AA_Right, 1)), which=:LM)
        ll_T, vv_T = partialeigen(decomp)
    else
        ll, vv = eigen(AA_Right)
        ll_T, vv_T = eigen(transpose(AA_Right))
    end
    vv_TT_vv = transpose(vv_T) * vv
    #     display(findall(abs.(vv_TT_vv) .> 1e-6))
    vv_T_perm = [argmax(abs.(vv_TT_vv[:, k])) for k in axes(vv_TT_vv, 2)]
    #     display(vv_T_invperm)
    #     vv_T_invperm = invperm(vv_T_perm)
    vv_T_scale = [vv_TT_vv[vv_T_perm[k], k] for k in axes(vv_TT_vv, 2)]
    vv_T = vv_T[:, vv_T_perm] * Diagonal(Skew_Dimension ./ vv_T_scale)

    #     vv_T = Diagonal(Skew_Dimension ./ vv_T_scale) * vv_T[:, vv_T_invperm]
    ll_T = ll_T[vv_T_perm]
    # diagnostics
    vv_TT_vv = transpose(vv_T) * vv
    #     display(findall(abs.(vv_TT_vv) .> 1e-6))
    #     offdiag = findall(abs.(vv_TT_vv - Diagonal(diag(vv_TT_vv))) .> 1e-6)
    #     min_row, max_row = extrema(x -> x[1], offdiag)
    #     min_col, max_col = extrema(x -> x[2], offdiag)
    #     display(vv_TT_vv[min_row:max_row, min_col:max_col])
    #     display(offdiag)
    #     display(vv_TT_vv[offdiag])
#     @show norm(vv_TT_vv - Diagonal(diag(vv_TT_vv))), diag(vv_TT_vv)
    #         @show vv_TT_vv
    #         @show log.(ll) ./ Time_Step
    ############ SORTING EIGENVALUES ############
    #     if by_residual
    #         # TODO YYt, YXt, XXt are not defined!
    #         @tullio num[k] := conj(vv[i,k]) * ( YYt[i,j] - ll[k] * YXt[j,i] - conj(ll[k]) * YXt[i,j] + ll[k] * conj(ll[k]) * XXt[i,j] ) * vv[j,k]
    #         @tullio den[k] := conj(vv[i,k]) * XXt[i,j] * vv[j,k]
    #         res = sqrt.(real.(num ./ den))
    #         pm = sortperm(res)  # by residual
    #     else
    pm = sortperm(importance.(ll), rev=true) # by magnitude
    #     end
    ############ finding clusters ############
    maxevals = min(3 * dims * Skew_Dimension, length(pm))
    clusters_pm, clrepr_pm, tv0 = Find_Eigenvector_Clusters(Skew_Dimension, ll[pm[1:maxevals]], vv[:, pm[1:maxevals]])
    tv = zero(tv0)
    tv[pm[1:maxevals]] .= tv0
    # mapping back to the original indices
    maxdims = min(dims + 1, length(clusters_pm))
    ord_clusters = sortperm(importance.(ll[pm[clrepr_pm]]), rev=true)[1:maxdims]
    clrepr = pm[clrepr_pm[ord_clusters]]
    clusters = [pm[cl] for cl in clusters_pm[ord_clusters]]
    # START SORTING: complex by frequencies, then real
    # real eigenvalues
    cl_r = clrepr[findall(imag.(ll[clrepr]) .== 0)]
    # complex eigenvalues
    cl_c = setdiff(clrepr, cl_r)
    #     cl_cs = cl_c[sortperm(freq_measure.(ll[cl_c]))]
    cl_cs = cl_c[sortperm(importance.(ll[cl_c]), rev=true)]
    cl_rs = cl_r[sortperm(importance.(ll[cl_r]), rev=true)]

    # putting into a matrix
    cl_cplx_p = cl_cs[findall(imag.(ll[cl_cs]) .> 0)]
    cl_cplx_n = cl_cs[findall(imag.(ll[cl_cs]) .< 0)]
    @show length(cl_cplx_p), length(cl_cplx_n)
    cl_pair = [findfirst(isapprox.(ll[cl_cplx_n], conj(a))) for a in ll[cl_cplx_p]]
    cl_cplx_fin_n = cl_cplx_n[cl_pair[findall(.!isnothing.(cl_pair))]]
    cl_cplx_fin_p = [cl_cplx_p[findfirst(isapprox.(real.(ll[cl_cplx_p]), a))] for a in real.(ll[cl_cplx_fin_n])]

    # doing the complex part
    cplx_vectors = deepcopy(vv[:, clrepr])
    cplx_vectors_T = deepcopy(vv_T[:, clrepr])
    cplx_values = ll[clrepr]

    # making it into real values
    real_vectors = zeros(size(vv, 1), 2 * length(cl_cplx_fin_n) + length(cl_rs))
    real_vectors[:, 1:2:(2*length(cl_cplx_fin_n))] .= real.(vv[:, cl_cplx_fin_n]) * 2
    real_vectors[:, 2:2:(2*length(cl_cplx_fin_n))] .= -imag.(vv[:, cl_cplx_fin_n]) * 2
    Real_Index_Start = 1 + 2 * length(cl_cplx_fin_n)
    real_vectors[:, Real_Index_Start:end] .= real.(vv[:, cl_rs])
    #
    real_values = zeros(2 * length(cl_cplx_fin_n) + length(cl_rs))
    real_values[1:2:(2*length(cl_cplx_fin_n))] .= real.(ll[cl_cplx_fin_n])
    real_values[2:2:(2*length(cl_cplx_fin_n))] .= -imag.(ll[cl_cplx_fin_n])
    real_values[Real_Index_Start:end] .= real.(ll[cl_rs])
    #
    real_vectors_T = zeros(size(vv_T, 1), 2 * length(cl_cplx_fin_n) + length(cl_rs))
    real_vectors_T[:, 1:2:(2*length(cl_cplx_fin_n))] .= real.(vv_T[:, cl_cplx_fin_n])
    real_vectors_T[:, 2:2:(2*length(cl_cplx_fin_n))] .= imag.(vv_T[:, cl_cplx_fin_n])
    real_vectors_T[:, Real_Index_Start:end] .= real.(vv_T[:, cl_rs])

    idx = zeros(eltype(clrepr), 2 * length(cl_cplx_fin_n) + length(cl_rs))
    idx[1:2:(2*length(cl_cplx_fin_n))] .= cl_cplx_fin_n
    idx[2:2:(2*length(cl_cplx_fin_n))] .= cl_cplx_fin_p
    idx[Real_Index_Start:end] .= cl_rs
    unmatched = setdiff(clrepr, idx)
    if isempty(unmatched)
        println("All eigenvalues matched.")
    else
        println("Unmatched complex eigenvalues")
        for cl in setdiff(clrepr, idx)
            println("  Eigenvalue ", cl, " = ", to_ode(ll[cl]) .* (1 / 2 / pi),
                " [Hz] ", " ==>> ", to_ode(ll[cl]), " [rad/s] tv = ", tv[cl])
            @show ll[clusters[findfirst(isequal(cl), clrepr)]]
        end
    end
    println("Trimmed eigenvalues")
    for cl in idx
        println("  Eigenvalue ", cl, " = ", to_ode(ll[cl]) .* (1 / 2 / pi),
            " [Hz] ", " ==>> ", to_ode(ll[cl]), " [rad/s] tv = ", tv[cl])
    end
    W_R = permutedims(reshape(real_vectors, Skew_Dimension, :, size(real_vectors, 2)), (2, 1, 3)) # * sqrt(Skew_Dimension)
    W_C = permutedims(reshape(cplx_vectors, Skew_Dimension, :, size(cplx_vectors, 2)), (2, 1, 3)) # * sqrt(Skew_Dimension)
    #
    V_R = permutedims(reshape(real_vectors_T, Skew_Dimension, :, size(real_vectors_T, 2)), (3, 1, 2)) # * sqrt(Skew_Dimension)
    V_C = permutedims(reshape(cplx_vectors_T, Skew_Dimension, :, size(cplx_vectors_T, 2)), (3, 1, 2)) # * sqrt(Skew_Dimension)
    return W_R, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values
end

function Bundle_Decomposition_By_Eigenvectors(BB, SH; Time_Step, sparse=false, dims=size(BB, 1))
    Skew_Dimension = size(BB, 2)
    AA_Right = Transfer_Operator_Right(BB, SH)
    W_R_SH, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values = Decompose_Eigenvectors(
        AA_Right, Skew_Dimension; Time_Step=Time_Step, sparse=sparse, dims=dims)
    @tullio W_R[i, k, j] := W_R_SH[i, l, j] * SH[k, l]
    V_R = zero(W_R)
    V_R_SH = zero(W_R)
    for k in axes(V_R, 2)
        V_R[:, k, :] .= inv(W_R[:, k, :])
        V_R_SH[:, k, :] .= inv(W_R_SH[:, k, :])
    end
    @tullio Lambda_P[i, k, r] := V_R_SH[i, k, j] * BB[j, k, l] * W_R[l, k, r]
    @show norm(Lambda_P .- mean(Lambda_P, dims=2))
    # remove the off diagonal zeros all just for testing
    Lambda_R = zero(Lambda_P[:, 1, :])
    Bundles = []
    for k in 1:2:Real_Index_Start-2
        push!(Bundles, k:k+1)
        Lambda_R[k:k+1, k:k+1] .= dropdims(mean(Lambda_P[k:k+1, :, k:k+1], dims=2), dims=2)
    end
    for k in Real_Index_Start:size(BB, 1)
        push!(Bundles, k:k)
        Lambda_R[k, k] = mean(Lambda_P[k, :, k])
    end
    # testing if it is constant
    @show norm(Lambda_P .- reshape(Lambda_R, size(Lambda_R, 1), 1, :))
    # testing over
    W_R_SH_ORTH = zero(W_R_SH)
    XX_SH = zero(W_R_SH)
    YY_SH = zero(W_R_SH)
    YY_SH_N = zero(W_R_SH)
    for Bundle in Bundles
        for k in axes(W_R, 2)
            F = svd(W_R_SH[:, k, Bundle])
            W_R_SH_ORTH[:, k, Bundle] .= F.U * F.Vt # [:, 1:length(Bundle)]
            # XX_SH[Bundle, k, Bundle] .= F.V * Diagonal(F.S) * F.Vt
            YY_SH_N[Bundle, k, Bundle] .= F.V * Diagonal(1 ./ F.S) * F.Vt
        end
        F = svd(reshape(view(YY_SH_N, Bundle, :, Bundle), length(Bundle), :)')
        YY_SH[Bundle, :, Bundle] .= reshape(F.V * F.U', length(Bundle), :, length(Bundle)) * sqrt(Skew_Dimension)
    end
    for Bundle in Bundles
        for k in axes(W_R, 2)
            XX_SH[Bundle, k, Bundle] .= inv(YY_SH[Bundle, k, Bundle])
        end
    end

    @tullio W_R_ORTH[i, k, j] := W_R_SH_ORTH[i, l, j] * SH[k, l]
    @tullio XX[i, k, j] := XX_SH[i, l, j] * SH[k, l]
    V_R_ORTH = zero(W_R)
    V_R_SH_ORTH = zero(W_R)
    YY = zero(XX)
    for k in axes(V_R, 2)
        V_R_ORTH[:, k, :] .= inv(W_R_ORTH[:, k, :])
        V_R_SH_ORTH[:, k, :] .= inv(W_R_SH_ORTH[:, k, :])
        YY[:, k, :] .= inv(XX[:, k, :])
    end
    @tullio Unreduced_Model[i, k, r] := V_R_SH_ORTH[i, k, j] * BB[j, k, l] * W_R_ORTH[l, k, r]
    @tullio Reduced_Model[i, k, r] := YY_SH[i, k, j] * Unreduced_Model[j, k, l] * XX[l, k, r]
    println("Bundle_Decomposition_By_Eigenvectors diagnostics:")
    # Reduced_Model = TR * YY_SH[i, k, j] * Unreduced_Model[j, k, l] * XX[l, k, r] * TR^-1
    @show norm(Reduced_Model .- reshape(Lambda_R, size(Lambda_R, 1), 1, :))
    @show norm(Reduced_Model .- mean(Reduced_Model, dims=2))
    @show norm(Unreduced_Model .- mean(Unreduced_Model, dims=2))
    Data_Encoder = V_R_ORTH
    Data_Decoder = W_R_ORTH
    Reduced_Encoder = YY
    Reduced_Decoder = XX
    return Unreduced_Model, Data_Encoder, Data_Decoder, Reduced_Model, Reduced_Decoder, Reduced_Encoder, Bundles
end

### TODO CREATE `AA_Right` SUCH THAT NO SHIFT IS NECESSARY in `WW`
function Decompose_Model_Right(BB, SH; Time_Step, sparse=false, dims=size(BB, 1))
    Skew_Dimension = size(SH, 1)
    AA_Right = Transfer_Operator_Right(BB, SH)
    W_R, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values = Decompose_Eigenvectors(
        AA_Right, Skew_Dimension; Time_Step=Time_Step, sparse=sparse, dims=dims)
    println("Decompose_Model_Right: Complex Bundle")
    # Complex vector bundle calculation
    @tullio W_Csh[i, k, j] := W_C[i, l, j] * SH[k, l]
    @tullio V_Csh[i, k, j] := V_C[i, l, j] * SH[k, l]
    V_I_C = similar(V_C)
    for k in axes(W_C, 2)
        TT = inv(V_C[:, k, :] * W_C[:, k, :])
        V_I_C[:, k, :] .= TT * V_C[:, k, :] #inv(W_C[:,k,:])
    end
    V_I_Csh = similar(V_Csh)
    for k in axes(W_Csh, 2)
        TT = inv(V_Csh[:, k, :] * W_Csh[:, k, :])
        V_I_Csh[:, k, :] .= TT * V_Csh[:, k, :] # inv(W_Csh[:,k,:])
    end
    @tullio Lambda_C[i, j, k] := V_I_C[i, k, r] * BB[r, k, s] * W_Csh[s, k, j]
    Lambda_Diagonal_C = diag(dropdims(mean(Lambda_C, dims=3), dims=3))
    # testing
    @tullio K[i, k, r] := W_C[i, k, j] * Lambda_C[j, l, k] * V_I_Csh[l, k, r]
    ll_err = norm(Lambda_C .-
                  reshape(diagm(Lambda_Diagonal_C), length(Lambda_Diagonal_C), length(Lambda_Diagonal_C), 1))
    if ll_err > 16 * sqrt(eps(1.0))
        println("Unexpected error in (complex) eigenvalue calculation = ", ll_err)
    end
    BB_err = norm(BB - K)
    if BB_err > 16 * sqrt(eps(1.0))
        println("Unexpected error in (complex) model decomposition = ", BB_err)
    end

    println("Decompose_Model_Right: Real Bundle")
    # Real vector bundle calculation
    V_I_R = similar(V_R)
    for k in axes(W_R, 2)
        TT = inv(V_R[:, k, :] * W_R[:, k, :])
        V_I_R[:, k, :] .= TT * V_R[:, k, :] # inv(W_R[:,k,:])
        #             display(V_R[:,k,:] * W_R[:,k,:])
    end
    @tullio W_Rsh[i, k, j] := W_R[i, l, j] * SH[k, l]
    @tullio V_Rsh[i, k, j] := V_R[i, l, j] * SH[k, l]
    V_I_Rsh = similar(V_Rsh)
    for k in axes(W_Rsh, 2)
        TT = inv(V_Rsh[:, k, :] * W_Rsh[:, k, :])
        V_I_Rsh[:, k, :] .= TT * V_Rsh[:, k, :] # inv(W_Rsh[:,k,:])
    end
    @tullio Lambda_R[i, j, k] := V_I_R[i, k, r] * BB[r, k, s] * W_Rsh[s, k, j]
    Lambda_Diagonal_R = dropdims(mean(Lambda_R, dims=3), dims=3)
    # testing
    @tullio K[i, k, r] := W_R[i, k, j] * Lambda_R[j, l, k] * V_I_Rsh[l, k, r]
    ll_err = norm(Lambda_R .- reshape(Lambda_Diagonal_R, size(Lambda_Diagonal_R, 1), size(Lambda_Diagonal_R, 2), 1))
    if ll_err > 16 * sqrt(eps(1.0))
        println("Unexpected error in (real) eigenvalue calculation = ", ll_err)
    end
    BB_err = norm(BB - K)
    if BB_err > 16 * sqrt(eps(1.0))
        println("Unexpected error in (real) model decomposition = ", BB_err)
    end
    # Reduced_Encoder = V_I_Rsh
    # Reduced_Decoder = W_Rsh
    return V_I_Rsh, W_Rsh, Lambda_Diagonal_R, V_I_Csh, W_Csh, Lambda_Diagonal_C, Real_Index_Start, V_I_C, W_C
end

function First_Spectrum_Point(BB, SH)
    Skew_Dimension = size(SH, 1)
    AA_Right = Transfer_Operator_Right(BB, SH)
    values, vectors = eigen(AA_Right)
    id = Minimum_Total_Variation_Of_Vectors(Skew_Dimension, vectors)
    return values[id]
end

function Decompose_Model_ODE(Jac, Generator; sparse=false, dims=size(Jac, 1))
    State_Dimension = size(Jac, 1)
    @assert State_Dimension == size(Jac, 2) "Decompose_Model_ODE: Jacobian is not square"
    Skew_Dimension = size(Jac, 3)
    Grid = Fourier_Grid(Skew_Dimension)
    Id_Skew = Diagonal(ones(State_Dimension))
    Big_Jac = zeros(Skew_Dimension, State_Dimension, Skew_Dimension, State_Dimension)
    for p in axes(Big_Jac, 1), q in axes(Big_Jac, 3)
        Big_Jac[p, :, q, :] .= Jac[:, :, p] * I[p, q] - Id_Skew * Generator[p, q]
    end
    Big_Jac_RS = reshape(Big_Jac, State_Dimension * Skew_Dimension, State_Dimension * Skew_Dimension)
    @show eigvals(Big_Jac_RS)
    W_R, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values = Decompose_Eigenvectors(
        Big_Jac_RS, Skew_Dimension; Time_Step=1.0, sparse=sparse, dims=dims, ODE=true)
    return W_R, W_C, V_R, V_C, Real_Index_Start, real_values, cplx_values
end

function Decompose_Diagonal_Right(BB, SH; Time_Step, sparse=false, dims=size(BB, 1))
    # V_I_Rsh,              :
    # W_Rsh,
    # Lambda_Diagonal_R,
    # V_I_Csh,
    # W_Csh,
    # Lambda_Diagonal_C,
    # Real_Index_Start,
    # V_I_C,
    # W_C
    return V_I_Rsh, W_Rsh, Lambda_Diagonal_R, V_I_Csh, W_Csh, Lambda_Diagonal_C, Real_Index_Start, V_I_C, W_C
end
