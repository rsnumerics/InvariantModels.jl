# SPDX-License-Identifier: EUPL-1.2

function meminfo_julia()
    # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
    # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
    @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes() / 2^20
    @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes() / 2^20
    return @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss() / 2^20
end

# replicates the structure of the ArrayPartition, but with single element arrays
function Replicate(x::ArrayPartition, Element_Type = Bool, Value = zero(Element_Type))
    return ArrayPartition(
        map(x -> x isa ArrayPartition ? Replicate(x, Element_Type, Value) : [Value], x.x),
    )
end

"""
    @enum Scaling_Type begin
        No_Scaling = 0
        Linear_Scaling = 1
        Quadratic_Scaling = 2
    end

Describes how to scale the loss function as a function of the distance from the steady state.
"""
@enum Scaling_Type begin
    No_Scaling = 0
    Linear_Scaling = 1
    Quadratic_Scaling = 2
end

struct Multi_Foliation{M, State_Dimension, Skew_Dimension} <: AbstractDecoratorManifold{ℂ}
    manifold::ProductManifold
    Indices::Any
    Scaling_Parameter::Any
    Scaling_Order::Scaling_Type
end

function Multi_Foliation(
        State_Dimension,
        Skew_Dimension,
        Trajectories::Int,
        Selection::NTuple{M, Vector{Int}},
        Model_Orders::NTuple{M, Int},
        Encoder_Orders::NTuple{M, Int};
        #     Linear_Model,
        #     Linear_Encoder,
        Scaling_Parameter = 2^(-4),
        Scaling_Order = No_Scaling,
        node_ratio = 0.8,
        leaf_ratio = 0.8,
        max_rank = 48,
        Linear_Type::NTuple{M, Encoder_Linear_Type} = ntuple(i -> Encoder_Array_Stiefel, M),
        Nonlinear_Type::NTuple{M, Encoder_Nonlinear_Type} = ntuple(
            i -> Encoder_Compressed_Latent_Linear,
            M,
        ),
    ) where {M}
    Full_Selection = vcat(Selection...)
    @assert sort(Full_Selection) == sort(unique(Full_Selection)) "Err 1"
    #     @assert length(Full_Selection) == State_Dimension
    Foliations = Array{Foliation, 1}(undef, M)
    Indices = Array{UnitRange, 1}(undef, M)
    Last_Index = 0
    for k in 1:M
        Latent_Dimension = length(Selection[k])
        Orthogonal_Indices = setdiff(1:State_Dimension, Selection[k])
        Foliations[k] = Foliation(
            State_Dimension,
            Latent_Dimension,
            Orthogonal_Indices,
            Skew_Dimension,
            Model_Orders[k],
            Encoder_Orders[k],
            Trajectories;
            node_ratio = node_ratio,
            leaf_ratio = leaf_ratio,
            max_rank = max_rank,
            Linear_Type = Linear_Type[k],
            Nonlinear_Type = Nonlinear_Type[k],
        )
        Indices[k] = range(Last_Index + 1, Last_Index + Latent_Dimension)
        Last_Index += Latent_Dimension
    end
    return Multi_Foliation{M, State_Dimension, Skew_Dimension}(
        ProductManifold(Foliations...),
        Indices,
        Scaling_Parameter,
        Scaling_Order,
    )
end

function Re_Target(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        Linear_Type::NTuple{M, Encoder_Linear_Type} = ntuple(i -> Encoder_Array_Stiefel, M),
        Nonlinear_Type::NTuple{M, Encoder_Nonlinear_Type} = ntuple(
            i -> Encoder_Compressed_Latent_Linear,
            M,
        ),
    ) where {M, State_Dimension, Skew_Dimension}
    Foliation_List = [MTF[k] for k in 1:length(MTF)]
    Encoder_List = [F[2] for F in Foliation_List]
    New_Encoder_List = []
    New_Foliation_List = []
    for k in 1:length(MTF)
        Foil = MTF[k]
        Model = Foil[1]
        Enc = Foil[2]
        pars = typeof(Enc).parameters
        New_Enc = QPEncoder{pars[1:4]..., Linear_Type[k], Nonlinear_Type[k], pars[7:end]...}(
            [getfield(Enc, p) for p in 1:nfields(Enc)]...,
        )
        println("Changing ", pars[5], " to ", Linear_Type[k], " ", Nonlinear_Type[k])
        push!(
            New_Foliation_List,
            typeof(Foil)(ManifoldsBase.ProductManifold(Model, New_Enc)),
        )
    end
    New_Manifold = ct.ManifoldsBase.ProductManifold(New_Foliation_List...)
    MMTF = typeof(MTF)(New_Manifold, MTF.Indices, MTF.Scaling_Parameter)
    return MMTF
end

"""
    Make_Similar(
        MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
        XTF,
        New_Trajectories::Integer,
    ) where {M,State_Dimension,Skew_Dimension}

Creates a new `Multi_Foliation` except that it will now have space for `New_Trajectories` initial conditions for the conjugate dynamics.
This is used to create a model to evaluate testing data that has different number of trajectories than the training data.
"""
function Make_Similar(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        XTF,
        New_Trajectories::Integer,
    ) where {M, State_Dimension, Skew_Dimension}
    Foliations = Array{Foliation, 1}(undef, M)
    Parameters = Array{ArrayPartition, 1}(undef, M)
    for k in 1:M
        MF, XF = Make_Similar(MTF[k], XTF.x[k], New_Trajectories)
        Foliations[k] = MF
        Parameters[k] = XF
    end
    return Multi_Foliation{M, State_Dimension, Skew_Dimension}(
            ProductManifold(Foliations...),
            deepcopy(MTF.Indices),
            deepcopy(MTF.Scaling_Parameter),
            deepcopy(MTF.Scaling_Order),
        ),
        ArrayPartition(Parameters...)
end

@inline Base.getindex(M::Multi_Foliation, i::Integer) = M.manifold[i]
@inline Base.length(M::Multi_Foliation) = length(M.manifold.manifolds)
@inline ManifoldsBase.decorated_manifold(M::Multi_Foliation) = M.manifold
@inline ManifoldsBase.get_forwarding_type(::Multi_Foliation, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::Multi_Foliation, ::Any, P::Type) = ManifoldsBase.SimpleForwardingType()

function Base.zero(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
    ) where {M, State_Dimension, Skew_Dimension}
    println("Multi_Foliation : zero()")
    XTF = ArrayPartition(map(zero, MTF.manifold.manifolds))
    for k in 1:M
        #         if Is_Oblique(MTF[k][2])
        #             @show XTF.x[k].x[2].x[2].x[1]
        #             @show XTF.x[k].x[2].x[2].x[2]
        #             XTF.x[k].x[2].x[2].x[1] .= 0
        #             XTF.x[k].x[2].x[2].x[2] .= 0
        #             XTF.x[k].x[2].x[2].x[1][MTF.Indices[k], :] .= Diagonal(I, length(MTF.Indices[k]))
        #             XTF.x[k].x[2].x[2].x[2][MTF.Indices[k], :, :] .= reshape(Diagonal(I, length(MTF.Indices[k])), length(MTF.Indices[k]), length(MTF.Indices[k]), 1)
        #         else
        #             @show XTF.x[k].x[2].x[2]
        #             XTF.x[k].x[2].x[2] .= 0
        #             XTF.x[k].x[2].x[2][MTF.Indices[k], :, :] .= reshape(Diagonal(I, length(MTF.Indices[k])), length(MTF.Indices[k]), length(MTF.Indices[k]), 1)
        #         end
    end
    return XTF
end

"""
    Slice(MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension}, XTF, Encoded_Slice) where {M,State_Dimension,Skew_Dimension}

In case the system is parameter dependent, this function creates an autonomous slice when the encoded parameter is `Encoded_Slice`.
"""
function Slice(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        XTF,
        Encoded_Slice,
    ) where {M, State_Dimension, Skew_Dimension}
    Foliations = Array{Foliation, 1}(undef, M)
    Parameters = Array{ArrayPartition, 1}(undef, M)
    for k in 1:M
        MF, XF = Slice(MTF[k], XTF.x[k], Encoded_Slice)
        Foliations[k] = MF
        Parameters[k] = XF
    end
    return Multi_Foliation{M, State_Dimension, 1}(
            ProductManifold(Foliations...),
            deepcopy(MTF.Indices),
            deepcopy(MTF.Scaling_Parameter),
            deepcopy(MTF.Scaling_Order),
        ),
        ArrayPartition(Parameters...)
end

# function Make_Orthogonal!(
#     MTF::Multi_Foliation{M,State_Dimension,Skew_Dimension},
#     XTF,
#     Index,
# ) where {M,State_Dimension,Skew_Dimension}
#     Latent_Dimension = Get_Latent_Dimension(MTF[Index])
#     Result = zeros(
#         eltype(XTF),
#         State_Dimension,
#         State_Dimension - Latent_Dimension,
#         Skew_Dimension,
#     )
#     Previous = XTF.x[Index].x[2].x[2]
#     it::Int = 0
#     for k = 1:length(MTF)
#         if k != Index
#             Local_Latent = Get_Latent_Dimension(MTF[k])
#             Linear_Part = view(Result, :, (it+1):(it+Local_Latent), :)
#             if Is_Mean_Stiefel(MTF[k][2])
#                 @tullio Linear_Part[i, j, k] = XTF.x[k].x[2].x[2][j, i, k]
#             else
#                 Linear_Part .= XTF.x[k].x[2].x[2]
#             end
#             it += Local_Latent
#         end
#     end
#     for k = 1:Skew_Dimension
#         Target = nullspace(transpose(view(Result, :, :, k)))
#         @show size(view(Result, :, :, k)), size(Previous[:, :, k]), size(Target)
#         F = svd(transpose(Previous[:, :, k]) * Target)
#         New_Projection = Target * transpose(F.U * F.Vt)
#         @show norm(New_Projection - view(Previous, :, :, k)),
#         norm(transpose(view(Result, :, :, k)) * view(Previous, :, :, k)),
#         norm(transpose(view(Result, :, :, k)) * New_Projection)
#         view(Previous, :, :, k) .= New_Projection
#     end
#     it = 0
#     for k = 1:length(MTF)
#         if k != Index
#             Local_Latent = Get_Latent_Dimension(MTF[k])
#             Linear_Part = view(Result, :, (it+1):(it+Local_Latent), :)
#             if Is_Mean_Stiefel(MTF[k][2])
#                 @tullio Linear_Part[i, j, k] = XTF.x[k].x[2].x[2][j, i, k]
#             else
#                 XTF.x[k].x[2].x[2] .= view(Result, :, (it+1):(it+Local_Latent), :)
#             end
#             it += Local_Latent
#         end
#     end
#     return nothing
# end
#
# function Make_Orthogonal!(MTF::Multi_Foliation, XTF)
#     Found_Orthogonal = false
#     for Index = 1:length(MTF)
#         if Is_Orthogonal(MTF[Index][2])
#             if Found_Orthogonal
#                 println("Make_Orthogonal: Too many orthogonal components")
#             end
#             Make_Orthogonal!(MTF, XTF, Index)
#             Found_Orthogonal = true
#         end
#     end
#     nothing
# end

function Compute_Scaling!(
        Scaling,
        Torus,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling_Parameter,
        Scaling_Order,
    )
    if Scaling_Order == No_Scaling
        Scaling .= 1
    elseif Scaling_Order == Linear_Scaling
        for k in axes(Data, 2)
            Norm_Squared = zero(eltype(Data))
            for i in axes(Torus, 1)
                Torus_Coordinate = zero(eltype(Data))
                for j in axes(Torus, 2)
                    Torus_Coordinate += Torus[i, j] * Encoded_Phase[j, k]
                end
                Norm_Squared += (Data[i, k] - Torus_Coordinate)^2
            end
            Norm_Lin = sqrt(Norm_Squared)
            # this is scaling by 1/x
            Inv_Scale = (
                Scaling_Parameter / 2 + Norm_Squared * Norm_Lin / (Scaling_Parameter^2) -
                    (1 / 2) * Norm_Squared * Norm_Squared / (Scaling_Parameter^3)
            )
            Scaling[k] = 1 / ifelse(Norm_Lin < Scaling_Parameter, Inv_Scale, Norm_Lin)
        end
    else
        for k in axes(Data, 2)
            Norm_Squared = zero(eltype(Data))
            for i in axes(Torus, 1)
                Torus_Coordinate = zero(eltype(Data))
                for j in axes(Torus, 2)
                    Torus_Coordinate += Torus[i, j] * Encoded_Phase[j, k]
                end
                Norm_Squared += (Data[i, k] - Torus_Coordinate)^2
            end
            Norm_Lin = sqrt(Norm_Squared)
            # this is scaling by 1/x^2
            Inv_Scale = (
                Scaling_Parameter^2 / 6 +
                    (4 / 3) * Norm_Squared * Norm_Lin / Scaling_Parameter -
                    (1 / 2) * Norm_Squared * Norm_Squared / (Scaling_Parameter^2)
            )
            Scaling[k] = 1 / ifelse(Norm_Lin < Scaling_Parameter, Inv_Scale, Norm_Squared)
        end
    end
    if Scaling_Order != No_Scaling
        for k in 1:(length(Index_List) - 1)
            NN = max(8, div(Index_List[k + 1] - Index_List[k], 48))
            for p in (1 + Index_List[k]):Index_List[k + 1]
                avg = 0.0
                To_Index = min(p + NN, Index_List[k + 1])
                for q in p:To_Index
                    avg += Scaling[q]
                end
                Scaling[p] = avg / (1 + abs(To_Index - p))
            end
        end
    end
    @show maximum(Scaling), minimum(Scaling)
    return nothing
end

struct Multi_Foliation_Cache{M, State_Dimension, Skew_Dimension}
    Parts::Any
    Torus::Any
    Scaling::Any
end

function Make_Cache(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        XTF,
        Index_List,
        Data,
        Encoded_Phase;
        Model = true,
        Shift = false,
        IC_Only = true,
        Scaling_Parameter = 2^(-2),
        Iterations = 32,
        Model_IC = false,
        Time_Step = 1.0,
    ) where {M, State_Dimension, Skew_Dimension}
    Scaling = zeros(eltype(Data), size(Data, 2))
    Torus = Find_Torus(MTF, XTF) # zeros(eltype(Data), State_Dimension, Skew_Dimension)
    Compute_Scaling!(
        Scaling,
        Torus,
        Index_List,
        Data,
        Encoded_Phase,
        Scaling_Parameter,
        MTF.Scaling_Order,
    )
    Parts = Array{Foliation_Cache, 1}(undef, M)
    Threads.@threads for Index in 1:M
        Parts[Index] = Make_Cache(
            MTF[Index],
            XTF.x[Index],
            Index_List,
            Data,
            Encoded_Phase,
            Scaling;
            Model = Model,
            Shift = Shift,
            IC_Only = IC_Only,
            Iterations = Iterations,
            Index = Index,
            Model_IC = Model_IC,
            Time_Step = Time_Step,
        )
    end
    return Multi_Foliation_Cache{M, State_Dimension, Skew_Dimension}(Parts, Torus, Scaling)
end

function Update_Scaling_All!(Cache, MTF, XTF, Index_List, Data, Encoded_Phase)
    Cache.Torus .= Find_Torus(MTF, XTF)
    println("Torus=", norm(Cache.Torus))
    Compute_Scaling!(
        Cache.Scaling,
        Cache.Torus,
        Index_List,
        Data,
        Encoded_Phase,
        MTF.Scaling_Parameter,
        MTF.Scaling_Order,
    )
    Threads.@threads for Part_Index in 1:length(MTF)
        Update_Cache!(
            Cache.Parts[Part_Index],
            MTF[Part_Index],
            XTF.x[Part_Index],
            Index_List,
            Data,
            Encoded_Phase,
            Cache.Scaling,
            (1,),
        )
    end
    return nothing
end

function Update_Cache!(Cache, MTF, XTF, Index_List, Data, Encoded_Phase, Component)
    # 1. update Encoder Cache
    # 2. Find Torus
    # 3. Update Scaling
    # 4. Update Model Cache
    Update_Cache!(
        Cache.Parts[Component[1]],
        MTF[Component[1]],
        XTF.x[Component[1]],
        Index_List,
        Data,
        Encoded_Phase,
        Cache.Scaling,
        Component[2:end],
        Model = false,
    )
    Update_Scaling!(Cache, MTF, XTF, Index_List, Data, Encoded_Phase, Component)
    return nothing
end

function Find_Torus(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        XTF;
        Iterations = 32,
    ) where {M, State_Dimension, Skew_Dimension}
    # performing Newton iteration on the encoders
    # The Jacobian is approximated using the linear parts of the encoders, because the root should be very close to the origin
    Project_Dimension = maximum(maximum.(MTF.Indices))
    QQ = zeros(Project_Dimension, State_Dimension, Skew_Dimension)
    #     QQ[1:Latent_Dimension, :, :] = permutedims(XTF.x[1].x[2].x[2], (2,1,3))
    #     QQ[Latent_Dimension + 1:end, :, :] = permutedims(XTF.x[2].x[2].x[2],(2,1,3))
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Residual = zeros(Project_Dimension, Skew_Dimension)
    Delta = zeros(State_Dimension, Skew_Dimension)
    Result = zeros(State_Dimension, Skew_Dimension)
    Pseudo_RTol = min(MTF.Scaling_Parameter, 2^(-4)) # should be very well conditioned
    Pseudo_Indices = zeros(Int, Skew_Dimension)
    Condition_Number = one(eltype(QQ))
    for it in 1:Iterations
        for k in 1:M
            @views Evaluate!(
                Residual[MTF.Indices[k], :],
                MTF[k][2],
                XTF.x[k].x[2],
                Result,
                Encoded_Phase,
            )
            @views Jacobian!(
                QQ[MTF.Indices[k], :, :],
                MTF[k][2],
                XTF.x[k].x[2],
                Result,
                Encoded_Phase,
            )
        end
        for p in axes(Delta, 2)
            F = svd(QQ[:, :, p])
            id_last = findlast(F.S .> F.S[1] * Pseudo_RTol)
            QQ_Pinv =
                F.V[:, 1:id_last] *
                Diagonal(1 ./ F.S[1:id_last]) *
                transpose(F.U[:, 1:id_last])
            Delta[:, p] .= QQ_Pinv * Residual[:, p]
            # logging
            Condition_p = minimum(F.S) / maximum(F.S)
            if Condition_Number > Condition_p
                Condition_Number = Condition_p
            end
            Pseudo_Indices[p] = id_last
        end
        Result .-= Delta
        #         display(Delta)
        #         @show norm(Delta), norm(Residual)
        if norm(Residual) < 16 * norm(QQ) * eps(eltype(Residual))
            break
        end
    end
    println(
        "Find_Torus: residual = ",
        norm(Residual),
        " resolved = ",
        Pseudo_Indices,
        " of ",
        State_Dimension,
        " with rtol = ",
        Pseudo_RTol,
        " vs ",
        Condition_Number,
    )
    return Result
end

# # this find the tangent of the first model if it is two dimensional
# NOTE : M_Model, X_Model are not needed, because they are part of MTF, XTF! ???
function Find_Tangent(
        MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension},
        XTF,
        M_Model,
        X_Model,
        Index,
        Torus,
        Beta_Grid;
        Select = [1; 2],
    ) where {M, State_Dimension, Skew_Dimension}
    # Latent_Tangent indices : 1 - dim, 2 - theta, 3 - beta
    #     Latent_Tangent, DR0, T0 = Find_Tangent(MTF[1][1], XTF.x[1].x[1], SH, Beta_Grid)
    #     _, Jac = Find_Torus(M_Model, X_Model, SH)
    Jac = view(X_Model.x[Part_WW], :, :, M_Model.Linear_Indices)
    Local_Latent = size(Jac, 1)
    #     @assert Local_Latent == 2 "Not a two-dimensional system"
    #     @show size(Jac)
    # finding the tangent bundle of the R. O. Model
    Latent_Tangent, DR0, T0 = Find_Tangent(M_Model, Jac, Select, Beta_Grid)
    #     @show size(Latent_Tangent)
    Latent_Extended =
        zeros(eltype(Latent_Tangent), State_Dimension, Skew_Dimension, length(Beta_Grid))
    #     @show size(Latent_Extended)
    Latent_Extended[MTF.Indices[Index], :, :] .= Latent_Tangent
    Encoded_Phase = diagm(ones(Skew_Dimension))
    Jac_Full = zeros(eltype(Torus), State_Dimension, State_Dimension, Skew_Dimension)
    for k in 1:M
        Jacobian!(
            view(Jac_Full, MTF.Indices[k], :, :),
            MTF[k][2],
            XTF.x[k].x[2],
            Torus,
            Encoded_Phase,
        )
    end
    #     display(Encoded_Phase)
    #     display(Jac_Full[:, :, 1])
    Full_Tangent = zero(Latent_Extended)
    #     # the Jacobian
    #     @show size(Jac_Full[1:Local_Latent, :, :])
    #     Jacobian!(view(Jac_Full, 1:Local_Latent, :, :), MTF[Index][2], XTF.x[Index].x[2], Torus, Encoded_Phase)
    #     it = Local_Latent
    #     for k in 1:length(MTF)
    #         if k != Index
    #             Part_Latent = typeof(MTF[k]).parameters[2]
    #             @views Jacobian!(Jac_Full[it+1:it+Part_Latent, :, :], MTF[k][2], XTF.x[k].x[2], Torus, Encoded_Phase)
    #             it += Part_Latent
    #         end
    #     end
    # the inversion
    for k in axes(Latent_Tangent, 2)
        IJAC = pinv(Jac_Full[:, :, k])
        ev = abs.(eigvals(Jac_Full[:, :, k]))
        #         @show k, minimum(ev), maximum(ev)
        for l in axes(Latent_Tangent, 3)
            Full_Tangent[:, k, l] .= IJAC * Latent_Extended[:, k, l]
        end
    end
    # Testing
    #     TST = reshape(zero(Latent_Tangent), Latent_Dimension, :)
    #     Data = zero(Full_Tangent)
    # #     @tullio Data[i, j, k] := Torus[i, j] + Full_Tangent[i, j, k]
    #     Phase = zeros(size(Encoded_Phase)..., length(Beta_Grid))
    #     for k in axes(Data, 3)
    #         Data[:, :, k] .= Torus + Full_Tangent[:, :, k]
    #         Phase[:, :, k] .= Encoded_Phase
    #     end
    #     Data_Full = reshape(Data, size(Data, 1), :)
    #     Phase_Full = reshape(Phase, size(Phase, 1), :)
    # #     X_Const = XTF.x[1].x[2].x[1]
    # #     X_Lin = XTF.x[1].x[2].x[2]
    # #     @tullio TST[i, k] = X_Const[i, q] * Phase_Full[q, k]
    # #     @tullio TST[i, k] += X_Lin[p, i, q] * Data_Full[p, k] * Phase_Full[q, k]
    #     Evaluate!(TST, MTF[1][2], XTF.x[1].x[2], Data_Full, Phase_Full)
    #     println("TST - Latent")
    #     display(reshape(TST - reshape(Latent_Tangent, Latent_Dimension, :), size(Latent_Tangent)...))
    #     println("TST")
    #     display(reshape(TST, size(Latent_Tangent)...))
    #     println("Latent")
    #     display(Latent_Tangent)
    return Full_Tangent, DR0, T0
end

# TODO testing must have its own MTF, because trajectory lengths and numbers are not the same!
# function Optimise_New!(MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension}, XTF,
#                    Index_List, Data, Encoded_Phase;
#                    Cache::Multi_Foliation_Cache=Make_Cache(MTF, XTF, Index_List, Data, Encoded_Phase),
#                    Radii = Replicate(XTF, Float64, 0.0), Counts=Replicate(XTF, Int),
#                    Steps = 128, Iterations = 32, Model_Iterations = Iterations, Gradient_Ratio = 2^(-7), Gradient_Stop = 2^(-29), Time_Step = 1.0,
#                    Picks = [Complete_Component_Index(XTF.x[k].x[2], (1,)) for k in 1:M], Name= "MTF-output", Test_Data = ()
#     ) where {M, State_Dimension, Skew_Dimension}
#     Train_Error = zeros(eltype(Data), size(Data, 2), M)
# #     @show size(Test_Data[1]), size(Test_Data[2]), size(Test_Data[3])
#     Test_MTF, Test_XTF = isempty(Test_Data) ? ([],[]) : Make_Similar(MTF, XTF, length(Test_Data[1])-1)
#     if !isempty(Test_Data)
#         Test_Index_List = Test_Data[1]
#         for Index in 1:M
#             for k in 1:length(Test_Index_List)-1 # Index_List
#                 Test_XTF.x[Index].x[1].x[Part_IC][:, k] .= 0 # Test_Cache.Parts[Index].Latent_Data[:, Test_Index_List[k]+1]
#             end
#         end
#     end
#     Test_Cache = isempty(Test_Data) ? [] : Make_Cache(Test_MTF, Test_XTF, Test_Data...; Model=false)
#     Test_Trace = isempty(Test_Data) ? [] : zeros(eltype(Data), 3, M, Steps) # mean, std, max
#     Test_Radii = isempty(Test_Data) ? [] : ones(eltype(Data), 3, M)
#     Test_Error = isempty(Test_Data) ? [] : zeros(eltype(Test_Data[2]), size(Test_Data[2], 2), M)
#     # counts the components of the Encoders
#     Component_Trace = Array{Any,2}(undef, M, Steps)
#     Loss_Trace = zeros(eltype(Data), M, Steps)
#     Error_Trace = zeros(eltype(Data), 3, M, Steps) # mean, std, max
#     Component_Pick = Array{Any,1}(undef, M)
#     Component_Pick .= Picks
#     for Step in 1:Steps
#         Threads.@threads for Index in 1:M
#             Component_Trace[Index, Step] = deepcopy(Component_Pick[Index])
#             Optimise_Next!(MTF[Index], XTF.x[Index], Index_List, Data, Encoded_Phase, Cache.Scaling, Step, view(Component_Pick, Index);
#                         Cache = Cache.Parts[Index],
#                         Radii = Radii.x[Index], Counts = Counts.x[Index],
#                         Iterations = Iterations, Model_Iterations = Model_Iterations,
#                         Gradient_Norm_Ratio = Gradient_Ratio, Gradient_Norm_Stop = Gradient_Stop, Time_Step = Time_Step, Index = Index)
#             Loss_Trace[Index, Step] = Loss(MTF[Index], XTF.x[Index], Index_List, Data, Encoded_Phase, Cache.Scaling, Cache = Cache.Parts[Index])
#             # Train_Error
#             Pointwise_Error!(view(Train_Error, :, Index), MTF[Index], XTF.x[Index], Index_List, Data, Encoded_Phase, Cache.Scaling, Cache = Cache.Parts[Index])
#             Error_Trace[1:2, Index, Step] .= mean_and_std(view(Train_Error, :, Index))
#             Error_Trace[3, Index, Step] = maximum(view(Train_Error, :, Index))
#             # Testing
#             if !isempty(Test_Data)
#                 if Component_Trace[Index, Step][1] == 1
#                     # preserve initial condition
#                     Test_XTF.x[Index].x[1].x[Part_WW] .= XTF.x[Index].x[1].x[Part_WW]
#                 else
#                     Get_Component(Test_XTF.x[Index], Component_Trace[Index, Step]) .= Get_Component(XTF.x[Index], Component_Trace[Index, Step])
#                 end
#                 Update_Cache!(Test_Cache.Parts[Index], MTF[Index], Test_XTF.x[Index], Test_Data..., Test_Cache.Scaling, Component_Trace[Index, Step]; Model=false)
#                 #
#                 if Component_Trace[Index, Step][1] == 1
#                     println("Updating Test Error ", Component_Trace[Index, Step])
#                     # might be a compressed model!
#                     Model_Phase = Get_Model_Phase(Test_MTF[Index], Test_Data[3])
#                     Maximum_Radius = 4 * sqrt(manifold_dimension(Test_MTF[Index][1]))
#                     Test_Radii[Index] = Optimise_IC_Full!(Test_MTF[Index][1], Test_XTF.x[Index].x[1],
#                                                         Test_Data[1], Test_Cache.Parts[Index].Latent_Data, Model_Phase, Test_Cache.Scaling;
#                                                         Cache=Test_Cache.Parts[Index].Model_Cache,
#                                                         Radius=Test_Radii[Index], Maximum_Radius=Maximum_Radius,
#                                                         Iterations=Iterations)
#                     if Test_Radii[Index] > Maximum_Radius / 8
#                         Test_Radii[Index] = Maximum_Radius / 8
#                     end
#                     Pointwise_Error!(view(Test_Error, :, Index), Test_MTF[Index], Test_XTF.x[Index], Test_Data..., Test_Cache.Scaling, Cache = Test_Cache.Parts[Index])
#                     Test_Trace[1:2, Index, Step] .= mean_and_std(view(Test_Error, :, Index))
#                     Test_Trace[3, Index, Step] = maximum(view(Test_Error, :, Index))
#                 end
#             end
#         end
#         To_Orthogonalise = false
#         for Index in 1:M
#             if Component_Trace[Index, Step] == (2, 2) # the linear part
#                 To_Orthogonalise = true
#             end
#         end
#         if To_Orthogonalise
#             Make_Orthogonal!(MTF, XTF)
#         end
#         Update_Scaling_All!(Cache, MTF, XTF, Index_List, Data, Encoded_Phase)
#         println("Saving into file: ", Name * ".bson")
#         if isempty(Test_Data)
#             JLSO.save(Name * ".bson", :MTF => MTF, :XTF => XTF,
#                       :Time_Step => Time_Step,
#                       :Loss_Trace => Loss_Trace, :Component_Trace => Component_Trace,
#                       :Error_Trace => Error_Trace, :Test_Trace => Test_Trace)
#         else
#             Compute_Scaling!(Test_Cache.Scaling, Cache.Torus, Test_Data[1], Test_Data[2], Test_Data[3], MTF.Scaling_Parameter, MTF.Scaling_Order)
#             JLSO.save(Name * ".bson", :MTF => MTF, :XTF => XTF, :Test_MTF => Test_MTF, :Test_XTF => Test_XTF,
#                       :Time_Step => Time_Step,
#                       :Loss_Trace => Loss_Trace, :Component_Trace => Component_Trace,
#                       :Error_Trace => Error_Trace, :Test_Trace => Test_Trace)
#         end
#         meminfo_julia()
#         GC.gc()
#     end
#     Training_Errors = zero(Train_Error)
#     Testing_Errors = isempty(Test_Data) ? [] : zero(Test_Error)
#     for Index in 1:M
#         Pointwise_Error!(view(Training_Errors, :, Index), MTF[Index], XTF.x[Index], Index_List, Data, Encoded_Phase, Cache.Scaling, Cache = Cache.Parts[Index])
#     end
#     if !isempty(Test_Data)
#         for Index in 1:M
#             Pointwise_Error!(view(Testing_Errors, :, Index), MTF[Index], XTF.x[Index], Test_Data..., Test_Cache.Scaling, Cache = Test_Cache.Parts[Index])
#         end
#     end
#     JLSO.save(Name * ".bson", :MTF => MTF, :XTF => XTF, :Time_Step => Time_Step,
#               :Loss_Trace => Loss_Trace, :Component_Trace => Component_Trace,
#               :Error_Trace => Error_Trace, :Test_Trace => Test_Trace,
#               :Training_Errors => Training_Errors, :Testing_Errors => Testing_Errors)
#     nothing
# end
