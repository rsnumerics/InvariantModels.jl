# SPDX-License-Identifier: EUPL-1.2

struct Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order} <:
    AbstractDecoratorManifold{ℂ}
    manifold::ProductManifold
end

function Get_Latent_Dimension(
        M::Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    ) where {State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    return Latent_Dimension
end

@inline Base.getindex(M::Foliation, i::Integer) = M.manifold[i]
@inline Base.length(M::Foliation) = length(M.manifold.manifolds)
@inline ManifoldsBase.decorated_manifold(M::Foliation) = M.manifold
@inline ManifoldsBase.get_forwarding_type(::Foliation, ::Any) = ManifoldsBase.SimpleForwardingType()
@inline ManifoldsBase.get_forwarding_type(::Foliation, ::Any, P::Type) = ManifoldsBase.SimpleForwardingType()

function Foliation(
        State_Dimension,
        Latent_Dimension,
        Orthogonal_Indices,
        Skew_Dimension,
        Model_Order,
        Encoder_Order,
        Trajectories;
        node_ratio = 1.0,
        leaf_ratio = 1.0,
        max_rank = 48,
        Linear_Type::Encoder_Linear_Type = Encoder_Array_Stiefel, Nonlinear_Type::Encoder_Nonlinear_Type = Encoder_Compressed_Latent_Linear
    )
    local Model
    if !Is_Model_Autonomous(Linear_Type)
        Model = MultiStep_Model(Latent_Dimension, Skew_Dimension, 1, Model_Order, Trajectories)
    else
        Model = MultiStep_Model(Latent_Dimension, 1, 1, Model_Order, Trajectories)
    end
    Encoder = QPEncoder(
        Latent_Dimension, State_Dimension, Orthogonal_Indices, Skew_Dimension, Encoder_Order,
        node_ratio = node_ratio, leaf_ratio = leaf_ratio, max_rank = max_rank,
        Linear_Type = Linear_Type, Nonlinear_Type = Nonlinear_Type
    )
    return Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}(
        ProductManifold(
            Model, Encoder
        )
    )
end

function Make_Similar(
        M::Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order},
        X,
        New_Trajectories::Integer
    ) where {State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    M_Model, X_Model = Make_Similar(M[1], X.x[1], New_Trajectories)
    M_Encoder = deepcopy(M[2])
    X_Encoder = deepcopy(X.x[2])
    return Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}(
            ProductManifold(M_Model, M_Encoder)
        ),
        ArrayPartition(X_Model, X_Encoder)
end

function Is_Model_Autonomous(MF::Foliation)
    return Is_Model_Autonomous(MF[2])
end

function Get_Model_Phase(MF::Foliation, Encoded_Phase)
    if Is_Model_Autonomous(MF)
        return ones(1, size(Encoded_Phase, 2))
    end
    return Encoded_Phase
end

function Base.zero(M::Stiefel{T, 𝔽}) where {T, 𝔽}
    return rand(M)
    #     if 𝔽 == ℂ
    #         return zeros(ComplexF64, representation_size(M)...)
    #     else
    #         return zeros(representation_size(M)...)
    #     end
end

function Base.zero(M::Foliation)
    return ArrayPartition(map(zero, M.manifold.manifolds))
end

function Slice(
        M::Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order},
        X,
        Encoded_Slice,
    ) where {State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    M_Model, X_Model = Slice(M[1], X.x[1], Encoded_Slice)
    M_Encoder, X_Encoder = Slice(M[2], X.x[2], Encoded_Slice)
    return Foliation{State_Dimension, Latent_Dimension, 1, Model_Order, Encoder_Order}(
            ProductManifold(M_Model, M_Encoder)
        ), ArrayPartition(X_Model, X_Encoder)
end

function Model_Part(MF::Foliation)
    return MF[1]
end

function Encoder_Part(MF::Foliation)
    return MF[2]
end

function Model_Point(XF::ArrayPartition)
    return XF.x[1]
end

function Encoder_Point(XF::ArrayPartition)
    return XF.x[2]
end

struct Foliation_Cache{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    Model_Cache::Any
    Encoder_Cache::ArrayPartition
    Model_Gradient::Any
    Latent_Data::Any #
    Latent_Delta::Any # same size as Latent_Data -> one part replaced with Delta
    Gradient_Raw::Any # same size as state -> for Hessian
    Gradient_Projected::Any # same size as state -> for finding next step
    Hessian_Delta::Any # Riemannian Hessian with Delta applied
end

# function Initialise(MF::Foliation, )

# Shift and Model_IC only matters if Model == false
function Make_Cache(
        MF::Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order},
        XF, Index_List, Data, Encoded_Phase, Scaling;
        Model = true, Shift = false, Iterations = 32, Index = 0, IC_Only = true,
        Model_IC = false,
        Time_Step = 1.0,
    ) where {State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    Latent_Data = zeros(eltype(Data), Latent_Dimension, size(Data, 2))
    Latent_Delta = zero(Latent_Data)
    Model_Phase = Get_Model_Phase(MF, Encoded_Phase)
    Encoder_Cache = Make_Cache(Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase)
    Evaluate!(Latent_Data, Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase, Cache = Encoder_Cache)
    if Model && Model_Order > 0
        # Linear = true means -> identify the full model, but only set the linear part
        From_Data!(Model_Part(MF), Model_Point(XF), Index_List, Latent_Data, Model_Phase, Scaling; Linear = true)
        AA_Right = Transfer_Operator_Right(Model_Point(XF).x[Part_WW][:, :, Model_Part(MF).Linear_Indices], Model_Part(MF).SH)
        ev = eigvals(AA_Right)
        @show mx = maximum(abs.(ev))
        if mx > 1
            Model_Point(XF).x[Part_WW] .*= (1 / mx)
        end
    elseif Shift
        SH = Find_Shift(Index_List, Latent_Data, Model_Phase)
        Model_Part(MF).SH .= SH
    end
    Model_Cache = Make_Cache(
        Model_Part(MF), Model_Point(XF), Index_List, Latent_Data, Model_Phase, Scaling, IC_Only = IC_Only
    )
    Model_Gradient = zeros(eltype(Data), size(Latent_Data)...)
    Cache = Foliation_Cache{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}(
        Model_Cache, Encoder_Cache,
        Model_Gradient,
        Latent_Data, Latent_Delta,
        zero(MF), zero(MF), zero(MF)
    )
    if Model_Order > 0
        if Model
            println("Optimising Model!")
            Optimise!(
                MF, XF, Index_List, Data, Encoded_Phase, Scaling, (1,);
                Cache = Cache, Iterations = Iterations, Time_Step = Time_Step, Index = Index, Step = 0
            )
            Pointwise_Gradient!(
                Cache.Model_Gradient, Model_Part(MF), Model_Point(XF), Index_List,
                Latent_Data, Model_Phase, Scaling, Cache = Model_Cache
            )
        elseif Model_IC
            println("Optimising Initial Conditions!")
            Model_Point(XF).x[Part_IC] .= Latent_Data[:, 1 .+ Index_List[1:(end - 1)]]
            Optimise_IC_Full!(
                Model_Part(MF), Model_Point(XF), Index_List, Latent_Data,
                Model_Phase, Scaling, Cache = Model_Cache, Iterations = Iterations
            )
        end
    end
    return Cache
end

# function Update_Cache_Model!(Cache::Foliation_Cache, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling)
#     Update_Cache_Data!(Cache.Model_Cache, Model_Part(MF), Model_Point(XF), Index_List, Cache.Latent_Data, Encoded_Phase, Scaling)
#     Pointwise_Gradient!(Cache.Model_Gradient, Model_Part(MF), Model_Point(XF), Index_List, Cache.Latent_Data, Encoded_Phase, Scaling, Cache=Cache.Model_Cache)
#     nothing
# end

function Update_Cache!(
        Cache::Foliation_Cache, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component; Model = true
    )
    if Component[1] != 1
        Update_Cache!(Cache.Encoder_Cache, Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase, Component[2:end])
        Evaluate!(
            Cache.Latent_Data, Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase, Cache = Cache.Encoder_Cache
        )
    end
    Model_Phase = Get_Model_Phase(MF, Encoded_Phase)
    if Model
        Update_Cache!(
            Cache.Model_Cache, Model_Part(MF), Model_Point(XF), Index_List, Cache.Latent_Data, Model_Phase, Scaling
        )
        Pointwise_Gradient!(
            Cache.Model_Gradient, Model_Part(MF), Model_Point(XF), Index_List,
            Cache.Latent_Data, Model_Phase, Scaling, Cache = Cache.Model_Cache
        )
    elseif Component[1] == 1
        Update_Cache!(
            Cache.Model_Cache, Model_Part(MF), Model_Point(XF), Index_List, Cache.Latent_Data, Model_Phase, Scaling
        )
    end
    return nothing
end

# function Relative_Error!(Error, MF :: Foliation, XF, Data, Encoded_Phase; Torus, Cache :: Foliation_Cache)
#     Residual = Cache.Model_Cache.Residual
#     for k in axes(Data, 2)
#         Norm_Squared = zero(eltype(Data))
#         for i in axes(Torus, 1)
#             Torus_Coordinate = zero(eltype(Data))
#             for j in axes(Torus, 2)
#                 Torus_Coordinate += Torus[i, j] * Encoded_Phase[j, k]
#             end
#             Norm_Squared += (Data[i, k] - Torus_Coordinate) ^ 2
#         end
#         Residual_Squared = zero(eltype(Data))
#         for j in axes(Residual, 1)
#             Residual_Squared += real(Residual[j, k] * conj(Residual[j, k]))
#         end
#         Error[k] = sqrt(Residual_Squared / Norm_Squared)
#     end
#     return nothing
# end

function Pointwise_Error!(
        Error, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling)
    )
    Model_Phase = Get_Model_Phase(MF, Encoded_Phase)
    return Pointwise_Error!(
        Error, Model_Part(MF), Model_Point(XF), Index_List,
        Cache.Latent_Data, Model_Phase, Scaling, Cache = Cache.Model_Cache
    )
end

function Loss(
        MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling)
    )
    Model_Phase = Get_Model_Phase(MF, Encoded_Phase)
    return Loss(
        Model_Part(MF), Model_Point(XF), Index_List, Cache.Latent_Data, Model_Phase, Scaling, Cache = Cache.Model_Cache
    )
end

# function Encoder_Gradient_Full!(Gradient, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component; Cache::Foliation_Cache=Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling))
#     L0_DF_parts!(Gradient, Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase, Component[2:end]; L0 = Cache.Model_Gradient, Cache=Cache.Encoder_Cache)
#     Get_Component(Cache.Gradient_Raw, Component) .= Gradient
#     return nothing
# end

function Gradient!(
        Gradient, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling)
    )
    Gradient_Raw = Get_Component(Cache.Gradient_Raw, Component)
    L0_DF_parts!(
        Gradient_Raw, Encoder_Part(MF), Encoder_Point(XF), Data, Encoded_Phase,
        Component[2:end]; L0 = Cache.Model_Gradient, Cache = Cache.Encoder_Cache
    )
    project!(Get_Component(MF, Component), Gradient, Get_Component(XF, Component), Gradient_Raw)
    return nothing
end

function Gradient_All!(
        Gradient, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling)
    )
    Start_Component = Complete_Component_Index(XF.x[2], (1,))
    Component = []
    push!(Component, Start_Component)
    while true
        Full_Component = (2, Component[1]...)
        Gradient!(
            Get_Component(Gradient, Full_Component), MF, XF, Index_List,
            Data, Encoded_Phase, Scaling, Full_Component, Cache = Cache
        )
        Component[1] = Next_Component(XF.x[2], Component[1])
        if Component[1] == Start_Component
            break
        end
    end
    return nothing
end

# function Encoder_Hessian_Full!(Hessian, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component; Cache::Foliation_Cache=Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling))
#     @time Scaled_Hessian_parts!(Hessian, Encoder_Part(MF), Encoder_Point(XF),
#                           Data, Encoded_Phase, Component[2:end],
#                           Scaling=Scaling, Cache=Cache.Encoder_Cache)
#     nothing
# end

# function Encoder_Hessian_From_Raw!(M, Hessian, p, Gradient_Raw, Hessian_Raw, Delta)
#     HH = zero(Gradient_Raw)
#     if length(size(Hessian_Raw)) == 2
#         @tullio HH[i] = Hessian_Raw[i,j] * Delta[j]
#     elseif length(size(Hessian_Raw)) == 4
#         @tullio HH[i,j] = Hessian_Raw[i,j,k,l] * Delta[k,l]
#     elseif length(size(Hessian_Raw)) == 6
#         @tullio HH[i,j,p] = Hessian_Raw[i,j,p,k,l,q] * Delta[k,l,q]
#     else
#         println("Error")
#     end
#     riemannian_Hessian!(M, Hessian, p, Gradient_Raw, HH, Delta)
#     return nothing
# end

function Hessian_Raw!(
        Hessian, Delta, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling)
    )
    L0_DF_DF_Delta_parts!(
        Hessian, Delta, Cache.Latent_Delta, Encoder_Part(MF), Encoder_Point(XF),
        Data, Encoded_Phase, Component[2:end],
        Scaling = Scaling, Cache = Cache.Encoder_Cache
    )
    return nothing
end

# this needs efficiency
# function Hessian!(Hessian, Delta, MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Component; Cache::Foliation_Cache=Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling))
#     L0_DF_DF_Delta_parts!(Hessian, Delta, Cache.Latent_Delta, Encoder_Part(MF), Encoder_Point(XF),
#                           Data, Encoded_Phase, Component[2:end],
#                           Scaling=Scaling, Cache=Cache.Encoder_Cache)
#     riemannian_Hessian!(Get_Component(MF, Component), Hessian, Get_Component(XF, Component), Get_Component(Cache.Gradient_Raw, Component), Hessian, Delta)
#     nothing
# end

mutable struct QPDebugTRO <: DebugAction
    print::Function
    t0::Any
    Time_Step::Any
    radius::Any
    Component::Any
    function QPDebugTRO(Time_Step, radius, Component, print::Function = print)
        return new(print, time(), Time_Step, [radius], Component)
    end
end

# we need to calculate the actual frequency to check progress at least at the linear level
function (d::QPDebugTRO)(mp, trs, i::Int)
    d.radius[1] = trs.trust_region_radius
    txt = (
        "    Step=$(d.Component[1]) | Foil=$(d.Component[2]) | O=$(d.Component[3:end]) -> $(i). " *
            @sprintf("time = %.1f[s] ", time() - d.t0) *
            @sprintf("F(x) = %.5e ", get_cost(mp, trs.p)) *
            @sprintf("G(x) = %.5e ", norm(get_manifold(mp), trs.p, trs.X)) *
            @sprintf("R = %.5e ", d.radius[1])
    )
    d.print(txt, repeat("\b", length(txt)), "\n")
    return GC.gc(true)
end

# 1. pickl with greatest gradient
# 2. pick with lowest count
# 3. if the same as previous, try the next one

function Pick_To_Optimise!(Proposed_Component, Component, Step, MF::QPEncoder, Gradient_Projected, Counts, NSEQ)
    # erase the previous gradient
    Get_Component(Gradient_Projected, Component) .= 0
    for it in 1:(NSEQ + 1)
        if mod(Step, NSEQ) == 0
            print("|s")
            # sequential component
            Proposed_Component[1], _ = Find_Minimum_Component(Counts)
            @show Proposed_Component[1]
        else
            Proposed_Component[1], val = Find_Maximum_Component(Gradient_Projected)
            print(@sprintf("|m=%.5e|", val))
        end
        if Proposed_Component[1] == Component # if the same as previous iteration, skip
            print("|S")
            continue
        end
        if manifold_dimension(Get_Component(MF, Proposed_Component[1])) == 0
            # not the root node AND matrix is square
            print("|F")
            continue
        end
        if Proposed_Component[1][1] == 2
            if Is_Linear_Fixed(MF)
                print("|C")
                Get_Component(Gradient_Projected, Proposed_Component[1]) .= 0
                continue
            end
        end
        break
    end
    return
end

# here p is the difference
function Loss_Quadratic!(
        MF::Foliation, XF, p, p_Diff, Loss_Input, Gradient, Hessian,
        Index_List, Data, Encoded_Phase, Scaling, Component; Cache
    )
    Get_Component(XF, Component) .= p
    Update_Cache!(Cache, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component)
    Loss_Int_Update = Loss(MF, XF, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
    return Loss_Int_Update
    #     p_Diff .= p
    #     p_Diff .-= Get_Component(XF, Component)
    #     Hessian_Raw!(Hessian, p_Diff, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component, Cache = Cache)
    #     return Loss + sum(Gradient .* p_Diff) + sum(Hessian .* p_Diff) / 2
end

function Gradient_Quadratic!(
        Gradient_p, MF::Foliation, XF, p, p_Diff, Gradient, Hessian,
        Index_List, Data, Encoded_Phase, Scaling, Component; Cache
    )
    #     p_Diff .= p
    #     p_Diff .-= Get_Component(XF, Component)
    #     Hessian_Raw!(Hessian, p_Diff, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component, Cache = Cache)
    #     project!(Get_Component(MF, Component), Gradient_p, Get_Component(XF, Component), Gradient + Hessian)
    # old style
    Get_Component(XF, Component) .= p
    Gradient!(Gradient_p, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component; Cache = Cache)
    return Gradient_p
end

function Hessian_Quadratic!(
        Hessian_p, MF::Foliation, XF, Delta, Hessian, Index_List, Data, Encoded_Phase, Scaling, Component; Cache
    )
    Hessian_Raw!(Hessian, Delta, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component, Cache = Cache)
    riemannian_Hessian!(
        Get_Component(MF, Component), Hessian_p, Get_Component(XF, Component),
        Get_Component(Cache.Gradient_Raw, Component), Hessian, Delta
    )
    return Hessian_p
end

function Optimise!(
        MF::Foliation{State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}, XF,
        Index_List, Data, Encoded_Phase, Scaling, Component;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling),
        Radius = 0, Iterations = 32, Model_Iterations = Iterations, Gradient_Norm_Ratio = 2^(-7),
        Gradient_Norm_Stop = 2^(-29), Time_Step = 1.0, Index = 1,
        Step = 1
    ) where {State_Dimension, Latent_Dimension, Skew_Dimension, Model_Order, Encoder_Order}
    Manifold = Get_Component(MF, Component)
    Point = Get_Component(XF, Component)
    Maximum_Radius = 4 * sqrt(manifold_dimension(Manifold))
    if Component == (1,)
        Model_Phase = Get_Model_Phase(MF, Encoded_Phase)
        if Radius == 0.0
            Trust_Radius = 1.0
        else
            Trust_Radius = Radius
        end
        if Model_Order > 0
            t0 = time()
            for it in 1:Model_Iterations
                Trust_Radius, M_Loss, M_Grad = Optimise!(
                    Manifold, Point, Index_List, Cache.Latent_Data, Model_Phase, Scaling,
                    Cache = Cache.Model_Cache, Radius = Trust_Radius, Maximum_Radius = Maximum_Radius
                )
                txt = (
                    "    Step=$(Step) | Model=$(Index) | -> $(it). " *
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
            # print frequencies!
            ev = First_Spectrum_Point(Point.x[Part_WW][:, :, Manifold.Linear_Indices], Manifold.SH)
            println(
                "    Model=$(Index) | -> " *
                    @sprintf(
                    "Frequency = %.5e[Hz], %.5e[rad/s] ", abs(angle(ev)) / Time_Step / (2 * pi),
                    abs(angle(ev)) / Time_Step
                ) *
                    @sprintf("Damping = %.5e[Hz] ", -log(abs(ev)) / abs(angle(ev)))
            )
            println()
        else
            Point .= 0
        end
        return Trust_Radius
    else
        Trust_Radius = 1.0
        Default_Trust_Radius = Maximum_Radius / 8
        if Radius == 0.0
            Trust_Radius = Default_Trust_Radius
        elseif Radius < Default_Trust_Radius * 2^(-6)
            Trust_Radius = Default_Trust_Radius * 2^(-6)
        else
            Trust_Radius = Radius
        end
        Debug_Object = QPDebugTRO(Time_Step, Trust_Radius, (Step, Index, Component...))
        Gradient!(
            Get_Component(Cache.Gradient_Projected, Component), MF, XF,
            Index_List, Data, Encoded_Phase, Scaling, Component, Cache = Cache
        )
        Gradient_Norm_Start = norm(Get_Component(Cache.Gradient_Projected, Component))
        #
        Hessian_Calls = 0
        p_Diff = zero(Point)
        Point_Save = deepcopy(Point)
        Loss_Int = Loss(MF, XF, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
        Gradient_Raw_Int = Get_Component(Cache.Gradient_Raw, Component)
        Hessian_Raw_Int = zero(Gradient_Raw_Int)
        Point_Update = trust_regions!(
            #         Point_Update = adaptive_regularization_with_cubics!(
            Manifold,
            (M, p) -> begin
                return Loss_Quadratic!(
                    MF::Foliation, XF, p, p_Diff, Loss_Int, Gradient_Raw_Int, Hessian_Raw_Int,
                    Index_List, Data, Encoded_Phase, Scaling, Component; Cache = Cache
                )
            end,
            (M, Gradient_p, p) -> begin
                Hessian_Calls = 0
                return Gradient_Quadratic!(
                    Gradient_p, MF, XF, p, p_Diff, Gradient_Raw_Int, Hessian_Raw_Int,
                    Index_List, Data, Encoded_Phase, Scaling, Component; Cache = Cache
                )
            end,
            (M, Hessian_p, p, Delta) -> begin # Y -> Hessian, X -> Delta
                Hessian_Calls += 1
                txt = @sprintf("H[%d]", Hessian_Calls)
                print(txt, repeat("\b", length(txt)))
                return Hessian_Quadratic!(
                    Hessian_p, MF, XF, Delta, Hessian_Raw_Int, Index_List,
                    Data, Encoded_Phase, Scaling, Component; Cache = Cache
                )
            end,
            deepcopy(Point),
            evaluation = InplaceEvaluation(),
            max_trust_region_radius = Maximum_Radius,
            trust_region_radius = Trust_Radius,
            stopping_criterion = StopWhenAny(
                StopWhenGradientNormLess(max(Gradient_Norm_Start * Gradient_Norm_Ratio, Gradient_Norm_Stop)),
                StopAfterIteration(Iterations)
            ),
            debug = Debug_Object
        )
        Point .= Point_Update
        Update_Cache!(Cache, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component)
        Loss_Int_Update = Loss(MF, XF, Index_List, Data, Encoded_Phase, Scaling, Cache = Cache)
        if Loss_Int < Loss_Int_Update
            Point .= Point_Save
            Update_Cache!(Cache, MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component)
            println(
                "Index=", Index, " REJECT ", Component, " Margin = ",
                @sprintf("%.5e", Loss_Int - Loss_Int_Update), " UPD = ", @sprintf("%.5e", Loss_Int_Update)
            )
        else
            println(
                "Index=", Index, " ACCEPT ", Component, " Gain = ",
                @sprintf("%.5e", Loss_Int - Loss_Int_Update), " UPD = ", @sprintf("%.5e", Loss_Int_Update)
            )
        end
        return Debug_Object.radius[1]
    end
end

function Optimise_Next!(
        MF::Foliation, XF, Index_List, Data, Encoded_Phase, Scaling, Step, Component_Pick;
        Cache::Foliation_Cache = Make_Cache(MF, XF, Index_List, Data, Encoded_Phase, Scaling),
        Radii = Replicate(XF, Float64, 0.0), Counts = Replicate(XF, Int),
        Iterations = 32, Model_Iterations = Iterations, Gradient_Norm_Ratio = 2^(-7),
        Gradient_Norm_Stop = 2^(-29), Time_Step = 1.0, Index = 1
    )
    NSEQ = length(MF[2]) + 1
    if mod(Step, NSEQ) == 1
        print("|g")
        Gradient_All!(Cache.Gradient_Projected, MF, XF, Index_List, Data, Encoded_Phase, Scaling; Cache = Cache)
    end

    Radius = Get_Component(Radii, Component_Pick[1])[1]
    Radius = Optimise!(
        MF, XF, Index_List, Data, Encoded_Phase, Scaling, Component_Pick[1];
        Cache = Cache,
        Radius = Radius,
        Iterations = Iterations, Model_Iterations = Model_Iterations,
        Gradient_Norm_Ratio = Gradient_Norm_Ratio,
        Gradient_Norm_Stop = Gradient_Norm_Stop,
        Time_Step = Time_Step, Index = Index, Step = Step
    )
    Get_Component(Radii, Component_Pick[1]) .= Radius
    if mod(Step, NSEQ) == 0
        Component_Pick[1] = (1,)
    else
        # proposes Encoder component
        Proposed_Component = Array{Any, 1}(undef, 1)
        Pick_To_Optimise!(
            Proposed_Component, Component_Pick[1][2:end], Step, MF[2], Cache.Gradient_Projected.x[2], Counts.x[2], NSEQ
        )
        Component_Pick[1] = (2, Proposed_Component[1]...)
    end

    return Radius
end
