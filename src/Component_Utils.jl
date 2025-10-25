function Get_Component(X::ArrayPartition, sel)
    if length(sel) == 1
        return X.x[sel[1]]
    elseif length(sel) == 2
        return X.x[sel[1]].x[sel[2]]
    elseif length(sel) == 3
        return X.x[sel[1]].x[sel[2]].x[sel[3]]
    elseif length(sel) == 4
        return X.x[sel[1]].x[sel[2]].x[sel[3]].x[sel[4]]
    else
        return zeros(0, 0)
    end
    nothing
end

function Get_Component(M::AbstractManifold, sel)
    if length(sel) == 0
        return M
    elseif length(sel) == 1
        return M[sel[1]]
    elseif length(sel) == 2
        return M[sel[1]][sel[2]]
    elseif length(sel) == 3
        return M[sel[1]][sel[2]][sel[3]]
    elseif length(sel) == 4
        return M[sel[1]][sel[2]][sel[3]][sel[4]]
    end
    nothing
end

# finds the first leaf from a node. If leaf returns same
function Complete_Component_Index(X::ArrayPartition, sel)
    if hasproperty(X.x[sel[end]], :x)
        return Complete_Component_Index(X.x[sel[end]], (sel..., 1))
    else
        return sel
    end
    nothing
end

# this is a very general algorithm that goes throuh all the nodes of X and wraps around at the end
function Next_Component(X::ArrayPartition, sel)
    #     println("ENTER: Next_Component")
    # returns the element indexed by sel, starting from the node X
    function Climb_Tree(X::ArrayPartition, sel)
        if isempty(sel)
            return X
        else
            return Climb_Tree(X.x[sel[1]], sel[2:end])
        end
        nothing
    end

    # finds the first non-end node as coming back down the tree
    # returns empty if at the very end
    function Last_Node(X::ArrayPartition, sel)
        if isempty(sel)
            return X, ()
        end
        a = Climb_Tree(X, sel[1:(end - 1)])
        if isa(a, ArrayPartition)
            if sel[end] < length(a.x)
                return a, sel
            else
                return Last_Node(X::ArrayPartition, sel[1:(end - 1)])
            end
        else
            println("Last_Node: node is not an ArrayPartition.")
            return a, (-1,)
        end
        nothing
    end

    a, selp = Last_Node(X, sel)
    if isempty(selp)
        #         println("LEAVE 1: Next_Component")
        return Complete_Component_Index(X, (1,))
    else
        #         println("LEAVE 2: Next_Component")
        return Complete_Component_Index(a, (selp[1:(end - 1)]..., selp[end] + 1))
    end
    nothing
end

function Count_Components(X::ArrayPartition)
    sel = Array{Any, 1}(undef, 1)
    sel_start = Complete_Component_Index(X, (1,))
    sel[1] = sel_start
    count = 1
    while true
        sel[1] = Next_Component(X, sel[1])
        if sel[1] == sel_start
            return count
        else
            count = count + 1
        end
    end
    return count
end

# exagerates S and U_0 (constant part of U)
function Find_Maximum_Component(X::ArrayPartition)
    sel = Array{Any, 1}(undef, 1)
    sel_start = Complete_Component_Index(X, (1,))
    sel[1] = sel_start
    selM = deepcopy(sel)
    valM = 0.0
    while true
        val = norm(Get_Component(X, sel[1]))
        # if the constant part, increase the importance
        # L_inf norm
        # val = maximum(abs.(Get_Component(X, sel[1])))
        if val > valM
            selM .= sel
            valM = val
        end
        # exit
        sel[1] = Next_Component(X, sel[1])
        if sel[1] == sel_start
            return selM[1], valM
        end
    end
    return selM[1], valM
end

function Find_Minimum_Component(X::ArrayPartition)
    sel = Array{Any, 1}(undef, 1)
    sel_start = Complete_Component_Index(X, (1,))
    sel[1] = sel_start
    selM = deepcopy(sel)
    valM = Inf
    while true
        val = norm(Get_Component(X, sel[1]))
        # if the constant part, increase the importance
        # L_inf norm
        # val = maximum(abs.(Get_Component(X, sel[1])))
        if val < valM
            selM .= sel
            valM = val
        end
        # exit
        sel[1] = Next_Component(X, sel[1])
        if sel[1] == sel_start
            return selM[1], valM
        end
    end
    return selM[1], valM
end

function Print_Components(X::ArrayPartition)
    print("(")
    sel = Array{Any, 1}(undef, 1)
    sel_start = Complete_Component_Index(X, (1,))
    sel[1] = sel_start
    while true
        print(Get_Component(X, sel[1])[1], ",")
        sel[1] = Next_Component(X, sel[1])
        if sel[1] == sel_start
            println("\b)")
            return nothing
        end
    end
    return nothing
end
