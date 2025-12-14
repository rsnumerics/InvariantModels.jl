# SPDX-License-Identifier: EUPL-1.2

struct Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension}
    MTF::Multi_Foliation{M, State_Dimension, Skew_Dimension}
    XTF::Any
    MTF_Cache::Multi_Foliation_Cache{M, State_Dimension, Skew_Dimension}
    Index_List::Any
    Data_Decomposed::Any
    Encoded_Phase::Any
    Name::Any
    Time_Step::Any
end

"""
    Multi_Foliation_Problem(
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Selection::NTuple{M, Vector{Int}},
        Model_Orders::NTuple{M, Int},
        Encoder_Orders::NTuple{M, Int},
        Unreduced_Model,
        Reduced_Model,
        Reduced_Encoder,
        SH,
        Initial_Iterations = 32,
        Scaling_Parameter = 2^(-4),
        Initial_Scaling_Parameter = 2^(-4),
        Scaling_Order = Linear_Scaling,
        node_ratio = 0.8,
        leaf_ratio = 0.8,
        max_rank = 48,
        Linear_Type::NTuple{M, Encoder_Linear_Type} = ntuple(i -> Encoder_Array_Stiefel, M),
        Nonlinear_Type::NTuple{M, Encoder_Nonlinear_Type} = ntuple(i -> Encoder_Compressed_Latent_Linear, M),
        Name = "MTF-output",
        Time_Step = 1.0,
    ) where {M}

Creates an object with multiple foliations that can be fitted to the data.

* `Index_List`, `Data_Decomposed` and `Encoded_Phase` The training data approximately in the cooredinates of the invariant vector bundles
    of the linear part of the system about the steady state.
    This transformation can be achieved by [`Decompose_Data`](@ref)
    and the decomposition can be obtained by [`Create_Linear_Decomposition`](@ref),
    which in turn is obtained by an initial linear approximation of the model using [`Estimate_Linear_Model`](@ref).
* `Selection` sets the vector bundles to be included in the calculated foliations.
* `Model_Orders` the polynomial orders of the conjugate maps for each foliation
* `Encoder_Orders` the polynomial orders of the conjugate maps for each foliation
* `Unreduced_Model` approximate linear model in the coordinate system of the data
* `Reduced_Model` approximate (autonomous) linear model in the coordinate system of `Reduced_Encoder`
* `Reduced_Encoder` a linear encoder that reduces the approximate linear map `Unreduced_Model` to an autonomous map.
    It can also be an identity, in which case the `Reduced_Model` and the `Unreduced_Model` are the same
* `SH` The forcing map, identified by [`Estimate_Linear_Model`](@ref)
* `Initial_Iterations` how many optimisation steps to take to refine the supplied linear model to the encoded data
* `Scaling_Parameter` the parameter the specifies data scaling
* `Initial_Scaling_Parameter` the parameter the specifies data scaling at the initial refinement of the model.
    Choosing it too small might result in diverging calculations.
* `Scaling_Order` whether to assign different important to various data points. This is must be one element of [`Scaling_Type`](@ref)
* `node_ratio` when calculating the rank of a component in the compressed tensor representation,
    by what ratio should we reduced the rank of the connecting parent node.
* `leaf_ratio` by how much are we allowed to reduce the rank of a leaf node from the dimeneionality of the tensor
* `max_rank` the maximum rank of any node in the compressed tensor representation
* `Linear_Type` this is a tuple that contains what restriction should apply to the linear part of the encoder.
    The values are taken from [`Encoder_Linear_Type`](@ref)
* `Nonlinear_Type` this is a tuple that contains what restriction should apply to the nonlinear part of the encoder.
    The values are taken from [`Encoder_Nonlinear_Type`](@ref)
* `Name` this string variabel is used for the name of the data file wher the foliation is periodically save to.
* `Time_Step` the sampling time step of the supplied data. This is used only to display frequency information during calulation,
    otherwise has no effect.
"""
function Multi_Foliation_Problem(
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Selection::NTuple{M, Vector{Int}},
        Model_Orders::NTuple{M, Int},
        Encoder_Orders::NTuple{M, Int},
        Unreduced_Model,
        Reduced_Model,
        Reduced_Encoder,
        SH,
        Initial_Iterations = 32,
        Scaling_Parameter = 2^(-4),
        Initial_Scaling_Parameter = 2^(-4),
        Scaling_Order = Linear_Scaling,
        node_ratio = 0.8,
        leaf_ratio = 0.8,
        max_rank = 48,
        Linear_Type::NTuple{M, Encoder_Linear_Type} = ntuple(i -> Encoder_Array_Stiefel, M),
        Nonlinear_Type::NTuple{M, Encoder_Nonlinear_Type} = ntuple(
            i -> Encoder_Compressed_Latent_Linear,
            M,
        ),
        Name = "MTF-output",
        Time_Step = 1.0,
        Train_Model = true,
    ) where {M}
    State_Dimension = size(Data_Decomposed, 1)
    Skew_Dimension = size(Encoded_Phase, 1)
    Trajectories = length(Index_List) - 1
    MTF = Multi_Foliation(
        State_Dimension,
        Skew_Dimension,
        Trajectories,
        Selection,
        Model_Orders,
        Encoder_Orders;
        Scaling_Parameter = Scaling_Parameter,
        Scaling_Order = Scaling_Order,
        node_ratio = node_ratio,
        leaf_ratio = leaf_ratio,
        max_rank = max_rank,
        Linear_Type = Linear_Type,
        Nonlinear_Type = Nonlinear_Type,
    )
    XTF = zero(MTF)
    # 1. Model 2. Model Shift 3. Encoder
    for k in eachindex(XTF.x)
        println("Setting up model $(k).")
        if size(XTF.x[k].x[1].x[Part_WW], 3) > 0
            if size(XTF.x[k].x[1].x[Part_WW], 2) == 1
                XTF.x[k].x[1].x[Part_WW][:, :, MTF[k][1].Linear_Indices] .=
                    mean(Reduced_Model[Selection[k], :, Selection[k]], dims = 2)
                MTF[k][1].SH .= 1
            else
                XTF.x[k].x[1].x[Part_WW][:, :, MTF[k][1].Linear_Indices] .=
                    Unreduced_Model[Selection[k], :, Selection[k]]
                MTF[k][1].SH .= SH
            end
        end
        # indices (input, output, shift) <- (output[Selection[k]], shift, input)
        @show size(XTF.x[k].x[2].x[2]), size(view(Reduced_Encoder, Selection[k], :, :))
        if typeof(MTF[k][2][2]) <: Mean_Flat_Stiefel
            XTF.x[k].x[2].x[2] .=
                permutedims(view(Reduced_Encoder, Selection[k], :, :), (1, 3, 2))
        end
        # otherwise, encoder is identity...
    end
    MTF_Cache = Make_Cache(
        MTF,
        XTF,
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Model = Train_Model,
        Shift = false,
        IC_Only = false,
        Scaling_Parameter = Initial_Scaling_Parameter,
        Iterations = Initial_Iterations,
        Model_IC = false,
        Time_Step = Time_Step,
    )

    return Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension}(
        MTF,
        XTF,
        MTF_Cache,
        Index_List,
        Data_Decomposed,
        Encoded_Phase,
        Name,
        Time_Step,
    )
end

# function Load_Multi_Foliation_Problem(
#     Index_List,
#     Data_Decomposed,
#     Encoded_Phase;
#     Name = "MTF-output",
# ) end

"""
    Multi_Foliation_Test_Problem(
        MTFP::Multi_Foliation_Problem{M,State_Dimension,Skew_Dimension},
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Initial_Scaling_Parameter = 2^(-4),
    ) where {M,State_Dimension,Skew_Dimension}

This creates a test problem with test data included.
The test data in `Index_List`, `Data_Decomposed` and `Encoded_Phase`. `MTFP` is a [`Multi_Foliation_Problem`](@ref).
`Initial_Scaling_Parameter` has the same meaning as in [`Multi_Foliation_Problem`](@ref) but applied to the testing data.
"""
function Multi_Foliation_Test_Problem(
        MTFP::Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension},
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Initial_Scaling_Parameter = 2^(-4),
    ) where {M, State_Dimension, Skew_Dimension}
    MTF, XTF = Make_Similar(MTFP.MTF, MTFP.XTF, length(Index_List) - 1)
    # zero the initial conditions
    for Index in 1:M
        for k in 1:(length(Index_List) - 1)
            XTF.x[Index].x[1].x[Part_IC][:, k] .= 0
        end
    end
    MTF_Cache = Make_Cache(
        MTF,
        XTF,
        Index_List,
        Data_Decomposed,
        Encoded_Phase;
        Model = false,
        Model_IC = true,
        Shift = false,
        IC_Only = true,
        Scaling_Parameter = Initial_Scaling_Parameter,
    )
    return Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension}(
        MTF,
        XTF,
        MTF_Cache,
        Index_List,
        Data_Decomposed,
        Encoded_Phase,
        MTFP.Name,
        MTFP.Time_Step,
    )
end

"""
    Optimise!(
        MTFP::Multi_Foliation_Problem{M,State_Dimension,Skew_Dimension},
        MTFP_Test::Union{Nothing,Multi_Foliation_Problem{M,State_Dimension,Skew_Dimension}} = nothing;
        Model_Iterations = 16,
        Encoder_Iterations = 8,
        Steps = 128,
        Gradient_Ratio = 2^(-7),
        Gradient_Stop = 2^(-29),
        Picks = [Complete_Component_Index(MTFP.XTF.x[k].x[2], (1,)) for k = 1:M],
    ) where {M,State_Dimension,Skew_Dimension}

Fits the set of foliations defined in `MTFP` to the given data.
* `MTFP`      a [`Multi_Foliation_Problem`](@ref)
* `MTFP_Test` a [`Multi_Foliation_Test_Problem`](@ref).
    The parameters of `MTFP_Test` are regularly synchronised with `MTFP`,
    except the initial condition of the latent model, which are fitted from the testing trajectories.
* `Model_Iterations` maximum number of optimisation steps taken when the conjugate dynamics is fitted to latent data.
    The optimisation is a version of the Levenberg-Marquardt method.
* `Encoder_Iterations` maximum number of optimisation steps for any component of the encoder. The optimisation technique is a manifold Newton trust region method.
* `Steps` the number of optimisation cycles for each foliation.
* `Gradient_Ratio` Optimisation of a component stops when the norm of the gradient has reduced by this factor
* `Gradient_Stop` Optimisation of a component stops when the norm of the gradient has reached this value
* `Picks` Used for testing the code, leave as is.
"""
function Optimise!(
        MTFP::Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension},
        MTFP_Test::Union{Nothing, Multi_Foliation_Problem{M, State_Dimension, Skew_Dimension}} = nothing;
        Model_Iterations = 16,
        Encoder_Iterations = 8,
        Steps = 128,
        Gradient_Ratio = 2^(-7),
        Gradient_Stop = 2^(-29),
        Picks = [Complete_Component_Index(MTFP.XTF.x[k].x[2], (1,)) for k in 1:M],
    ) where {M, State_Dimension, Skew_Dimension}
    Index_List = MTFP.Index_List
    Data_Decomposed = MTFP.Data_Decomposed
    Encoded_Phase = MTFP.Encoded_Phase
    MTF = MTFP.MTF
    XTF = MTFP.XTF
    Cache = MTFP.MTF_Cache
    # when the testing error is minimal...
    XTF_Best = zero(MTF)
    # training traces
    Component_Trace = Array{Any, 2}((undef), M, Steps)
    Component_Trace .= (0,)
    Train_Error = zeros(eltype(Data_Decomposed), size(Data_Decomposed, 2), M)
    Train_Loss_Trace = zeros(eltype(Data_Decomposed), M, Steps)
    Train_Error_Trace = zeros(eltype(Data_Decomposed), 3, M, Steps) # mean, std, max
    #
    Radii = Replicate(MTFP.XTF, Float64, 0.0)
    Counts = Replicate(MTFP.XTF, Int)
    # testing traces
    Test_Error =
        isnothing(MTFP_Test) ? [] :
        zeros(eltype(MTFP_Test.Data_Decomposed), size(MTFP_Test.Data_Decomposed, 2), M)
    Test_Error_Trace =
        isnothing(MTFP_Test) ? [] : zeros(eltype(MTFP_Test.Data_Decomposed), 3, M, Steps) # mean, std, max
    Test_Loss_Trace =
        isnothing(MTFP_Test) ? [] : zeros(eltype(MTFP_Test.Data_Decomposed), M, Steps)
    Test_Radii = isnothing(MTFP_Test) ? [] : ones(eltype(MTFP_Test.Data_Decomposed), 3, M)
    # counts the components of the Encoders
    Component_Pick = Array{Any, 1}(undef, M)
    Component_Pick .= Picks
    for Step in 1:Steps
        Threads.@threads for Index in 1:M
            Component_Trace[Index, Step] = deepcopy(Component_Pick[Index])
            Optimise_Next!(
                MTF[Index],
                XTF.x[Index],
                Index_List,
                Data_Decomposed,
                Encoded_Phase,
                Cache.Scaling,
                Step,
                view(Component_Pick, Index);
                Cache = Cache.Parts[Index],
                Radii = Radii.x[Index],
                Counts = Counts.x[Index],
                Iterations = Encoder_Iterations,
                Model_Iterations = Model_Iterations,
                Gradient_Norm_Ratio = Gradient_Ratio,
                Gradient_Norm_Stop = Gradient_Stop,
                Time_Step = MTFP.Time_Step,
                Index = Index,
            )
            Train_Loss_Trace[Index, Step] = Loss(
                MTF[Index],
                XTF.x[Index],
                Index_List,
                Data_Decomposed,
                Encoded_Phase,
                Cache.Scaling,
                Cache = Cache.Parts[Index],
            )
            # Train_Error
            Pointwise_Error!(
                view(Train_Error, :, Index),
                MTF[Index],
                XTF.x[Index],
                Index_List,
                Data_Decomposed,
                Encoded_Phase,
                Cache.Scaling,
                Cache = Cache.Parts[Index],
            )
            Train_Error_Trace[1:2, Index, Step] .= mean_and_std(view(Train_Error, :, Index))
            Train_Error_Trace[3, Index, Step] = maximum(view(Train_Error, :, Index))
            # ----------------- BEGIN TESTING ------------------------#
            if !isnothing(MTFP_Test)
                Current_Component = Component_Trace[Index, Step]
                Test_Index_List = MTFP_Test.Index_List
                Test_Data_Decomposed = MTFP_Test.Data_Decomposed
                Test_Encoded_Phase = MTFP_Test.Encoded_Phase
                Test_MTF = MTFP_Test.MTF
                Test_XTF = MTFP_Test.XTF
                Test_Cache = MTFP_Test.MTF_Cache

                if Current_Component[1] == 1
                    # preserve initial condition
                    Test_XTF.x[Index].x[1].x[Part_WW] .= XTF.x[Index].x[1].x[Part_WW]
                else
                    Get_Component(Test_XTF.x[Index], Current_Component) .=
                        Get_Component(XTF.x[Index], Current_Component)
                end
                Update_Cache!(
                    Test_Cache.Parts[Index],
                    Test_MTF[Index],
                    Test_XTF.x[Index],
                    Test_Index_List,
                    Test_Data_Decomposed,
                    Test_Encoded_Phase,
                    Test_Cache.Scaling,
                    Current_Component;
                    Model = false,
                )
                #
                if Current_Component[1] == 1
                    println("Updating Test Error ", Current_Component)
                    # might be a compressed model!
                    Model_Phase = Get_Model_Phase(Test_MTF[Index], Test_Encoded_Phase)
                    Maximum_Radius = 4 * sqrt(manifold_dimension(Test_MTF[Index][1]))
                    Test_Radii[Index] = Optimise_IC_Full!(
                        Test_MTF[Index][1],
                        Test_XTF.x[Index].x[1],
                        Test_Index_List,
                        Test_Cache.Parts[Index].Latent_Data,
                        Model_Phase,
                        Test_Cache.Scaling;
                        Cache = Test_Cache.Parts[Index].Model_Cache,
                        Radius = Test_Radii[Index],
                        Maximum_Radius = Maximum_Radius,
                        Iterations = Model_Iterations,
                        Model_Index = Index,
                    )
                    if Test_Radii[Index] > Maximum_Radius / 8
                        Test_Radii[Index] = Maximum_Radius / 8
                    end
                    if Step > 1
                        Test_Loss_Trace[Index, Step] = Loss(
                            Test_MTF[Index],
                            Test_XTF.x[Index],
                            Index_List,
                            Data_Decomposed,
                            Encoded_Phase,
                            Test_Cache.Scaling,
                            Cache = Test_Cache.Parts[Index],
                        )
                    end
                    Pointwise_Error!(
                        view(Test_Error, :, Index),
                        Test_MTF[Index],
                        Test_XTF.x[Index],
                        Test_Index_List,
                        Test_Data_Decomposed,
                        Test_Encoded_Phase,
                        Test_Cache.Scaling,
                        Cache = Test_Cache.Parts[Index],
                    )
                    Test_Error_Trace[1:2, Index, Step] .=
                        mean_and_std(view(Test_Error, :, Index))
                    Test_Error_Trace[3, Index, Step] = maximum(view(Test_Error, :, Index))
                    #
                    nz_index = findall(!iszero, Test_Loss_Trace[Index, :])
                    if !isempty(nz_index)
                        nz_min = argmin(view(Test_Loss_Trace, Index, nz_index))
                        if nz_min == length(nz_index)
                            XTF_Best.x[Index] .= XTF.x[Index]
                        else
                            println(
                                "==>> Test result has NOT improved!",
                                @sprintf(
                                    " Best = %.5e",
                                    Test_Loss_Trace[Index, nz_index[nz_min]]
                                ),
                                @sprintf(" current = %.5e", Test_Loss_Trace[Index, Step])
                            )
                        end
                    end
                    #
                end
            end
            # ----------------- END TESTING ------------------------#
        end
        #         To_Orthogonalise = false
        #         for Index = 1:M
        #             if Component_Trace[Index, Step] == (2, 2) # the linear part
        #                 To_Orthogonalise = true
        #             end
        #         end
        #         if To_Orthogonalise
        #             Make_Orthogonal!(MTF, XTF)
        #         end
        Update_Scaling_All!(Cache, MTF, XTF, Index_List, Data_Decomposed, Encoded_Phase)
        println("Saving into file: ", MTFP.Name * ".bson")
        if !isnothing(MTFP_Test)
            Test_Index_List = MTFP_Test.Index_List
            Test_Data_Decomposed = MTFP_Test.Data_Decomposed
            Test_Encoded_Phase = MTFP_Test.Encoded_Phase
            Test_MTF = MTFP_Test.MTF
            Test_XTF = MTFP_Test.XTF
            Test_Cache = MTFP_Test.MTF_Cache
            #
            Compute_Scaling!(
                Test_Cache.Scaling,
                Cache.Torus,
                Test_Index_List,
                Test_Data_Decomposed,
                Test_Encoded_Phase,
                MTF.Scaling_Parameter,
                MTF.Scaling_Order,
            )
            JLSO.save(
                MTFP.Name * ".bson",
                :MTF => MTF,
                :XTF => XTF,
                :Test_MTF => Test_MTF,
                :Test_XTF => Test_XTF,
                :Best_XTF => XTF_Best,
                :Time_Step => MTFP.Time_Step,
                :Component_Trace => Component_Trace,
                :Train_Loss_Trace => Train_Loss_Trace,
                :Train_Error_Trace => Train_Error_Trace,
                :Test_Loss_Trace => Test_Loss_Trace,
                :Test_Error_Trace => Test_Error_Trace,
                format = :bson,
                compression = :gzip_fastest,
            )
        else
            JLSO.save(
                MTFP.Name * ".bson",
                :MTF => MTF,
                :XTF => XTF,
                :Time_Step => MTFP.Time_Step,
                :Component_Trace => Component_Trace,
                :Train_Loss_Trace => Train_Loss_Trace,
                :Train_Error_Trace => Train_Error_Trace,
                format = :bson,
                compression = :gzip_fastest,
            )
        end
        meminfo_julia()
        GC.gc()
    end
    return nothing
end
