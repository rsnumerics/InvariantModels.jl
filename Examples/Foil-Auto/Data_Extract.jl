using JLSO, StatsBase, Interpolations, LinearAlgebra

function Extract_From_Raw(traces_raw, idc)
    labels = unique([x[1] for x in traces_raw])
    @show labels
    bb = [filter(x -> x[1] == lb, traces_raw) for lb in labels]

    traces = []
    times = []
    for cc in bb
        push!(traces, vcat([c[4][idc, :] for c in cc]...))
        push!(times, cc[1][5] .- cc[1][5][1])
    end
    return traces, times
end

Traces_Raw_Master = JLSO.load("dw-auto-master.bson")
Traces_Master, Times_Master = Extract_From_Raw(Traces_Raw_Master[:traces], [1])
Traces_Raw_Slave = JLSO.load("dw-auto-slave.bson")
Traces_Slave, Times_Slave = Extract_From_Raw(Traces_Raw_Slave[:traces], [1])

@show [size(a) for a in Traces_Master]
@show [size(a) for a in Traces_Slave]

@show Time_Step = mean([(tm[end] - tm[1]) / (length(tm)-1) for tm in Times_Master])
@show 1e9 / Time_Step, "mean fps"

function vv_cat(tp)
    sz = minimum([size(k, 2) for k in tp])
    return vcat([k[:, 1:sz] for k in tp]...)
end

Traces_Full_Pre = [vv_cat(k) for k in zip(Traces_Master, Traces_Slave)]
Times_Itp = [range(start=tm[1], stop=tm[size(sig, 2)], step=Time_Step) for (sig, tm) in zip(Traces_Full_Pre, Times_Master)]
Traces_Full = [zeros(eltype(tr), size(tr, 1), length(tm)) for (tr, tm) in zip(Traces_Full_Pre, Times_Itp)]

for k in eachindex(Traces_Full)
    tr_pre = Traces_Full_Pre[k]
    tr = Traces_Full[k]
    tm_pre = Times_Master[k]
    tm = Times_Itp[k]
    for p in axes(tr, 1)
        itp = linear_interpolation(tm_pre[1:size(tr_pre, 2)], tr_pre[p, :])
        tr[p, :] .= itp.(tm)
    end
end

Start_Delay = 25
Impacts_At = []
for tr in Traces_Full
    mx_all = vec(maximum(abs.(tr), dims=1))
    mx = maximum(mx_all)
    push!(Impacts_At, findfirst(mx_all .> 0.1 * mx) + Start_Delay)
end

# removing steady state
for k in eachindex(Traces_Full)
    Steady_State = mean(Traces_Full[k][:, end-1000:end], dims=2)
    Traces_Full[k] .-= Steady_State
end

Signal_Length = 8000
Traces = [Traces_Full[k][:, Impacts_At[k]:min(Impacts_At[k]+Signal_Length-1, size(Traces_Full[k], 2))] for k in eachindex(Impacts_At)]

JLSO.save("Data_Traces.bson", :Traces => Traces, :Time_Step => Time_Step / 1e9) # 384.3406413156464 [fps]
