module de
using JLSO, StatsBase, Interpolations, LinearAlgebra, FFTW

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

function sigfilt(sig, low, high, length)
    ff = fft(sig)
    gg = zeros(eltype(ff), length)
    gg[1:div(length, 2)] .= ff[1:div(length, 2)]
    gg[end-div(length, 2):end] .= ff[end-div(length, 2):end]
    gg[1:low] .= 0
    gg[high+2:end-high] .= 0
    gg[end-low+2:end] .= 0
    return real.(ifft(gg))
end

Traces_Raw_Master = JLSO.load("dw-forced-master-fix.bson")
Traces_Master, Times_Master = Extract_From_Raw(Traces_Raw_Master[:traces], [1])
Traces_Raw_Slave = JLSO.load("dw-forced-slave-fix.bson")
Traces_Slave, Times_Slave = Extract_From_Raw(Traces_Raw_Slave[:traces], [1])

Phases_Angle_Pre = JLSO.load("dw-forced-phase-angles.bson")[:phases]
# making it uniform
Phases_Angle = [range(ph[1], ph[end], length=length(ph)) for ph in Phases_Angle_Pre]
@show [maximum(abs.(Phases_Angle[k] .- Phases_Angle_Pre[k])) for k in eachindex(Phases_Angle)]

@show [size(a) for a in Traces_Master]
@show [size(a) for a in Traces_Slave]
@show [size(a) for a in Phases_Angle]

function vv_cat(tp)
    sz = minimum([size(k, 2) for k in tp])
    return vcat([k[:, 1:sz] for k in tp]...)
end

# putting the master / slave together
Traces_Full_Pre = [vv_cat(k) for k in zip(Traces_Master, Traces_Slave)]

# filling in the NaNs
for sig in Traces_Full_Pre
    id = findall(isnan.(sig))
    for k in id
        sig[k] = (sig[k[1], k[2]-1] + sig[k[1], k[2]+1]) / 2
    end
end

# sampling period
@show Time_Step = mean([(tm[end] - tm[1]) / (length(tm)-1) for tm in Times_Master]) / 1e9 # measured in nano seconds -> convert to seconds

# removing steady state
for k in eachindex(Traces_Full_Pre)
    start = argmin(abs.(Phases_Angle[k] .- (Phases_Angle[k][end] - 70 * pi)))
    Steady_State = mean(Traces_Full_Pre[k][:, start:end], dims=2)
    Traces_Full_Pre[k] .-= Steady_State
end

# finding impact positions
Start_Delay = 25
Impacts_At = []
for tr in Traces_Full_Pre
    mx_all = vec(maximum(abs.(tr), dims=1))
    mx = maximum(mx_all)
    push!(Impacts_At, findfirst(mx_all .> 0.5 * mx) .+ Start_Delay)
end

Signal_Length = 8000
Times_Cut = [(Impacts_At[k] + Start_Delay):(Impacts_At[k] + Start_Delay + Signal_Length - 1) for k in eachindex(Impacts_At)]
Traces_Full_Cut = [sig[:,cut] for (sig, cut) in zip(Traces_Full_Pre, Times_Cut)]
Phases_Angle_Cut = [sig[cut] for (sig, cut) in zip(Phases_Angle, Times_Cut)]

Max_FFT_Delay = minimum(Impacts_At) - 1
Max_FFT_Length = minimum([length(tm) - Max_FFT_Delay for tm in Times_Master])
Max_FFT_Length = Max_FFT_Length - mod(Max_FFT_Length + 1, 2) # making it odd

Skew_Dimension = 17

Start_Freq = 0 # Hz
Stop_Freq = 80 # Hz

Start_FFT = max(1, round(Int, Start_Freq * Time_Step * Max_FFT_Length))
Stop_FFT = round(Int, Stop_Freq * Time_Step * Max_FFT_Length)

FFT_Interpolation_Length = length(range(start=Phases_Angle[1][Impacts_At[1] - Max_FFT_Delay], stop=Phases_Angle[1][Impacts_At[1] - Max_FFT_Delay + Max_FFT_Length - 1], step=2 * pi / Skew_Dimension))

# The Hann window
Window = sin.(range(0, pi, length=Max_FFT_Length)) .^ 2
Window_Post = sin.(range(0, pi, length=FFT_Interpolation_Length)) .^ 2

Impact_At_FFT = [max(1, round(Int, Max_FFT_Delay * (FFT_Interpolation_Length / Max_FFT_Length))) for imp in Impacts_At]
Signal_Length_FFT = round(Int, Signal_Length * (FFT_Interpolation_Length / Max_FFT_Length))

Times_Full = [(Impacts_At[k] - Max_FFT_Delay):(Impacts_At[k] - Max_FFT_Delay + Max_FFT_Length - 1) for k in eachindex(Impacts_At)]
Traces_Full = [ vcat([
                        reshape(sigfilt(Traces_Full_Pre[k][p, Times_Full[k]] .* Window, Start_FFT, Stop_FFT, FFT_Interpolation_Length) * (FFT_Interpolation_Length / Max_FFT_Length) ./ Window_Post, 1, :)
                        for p in axes(Traces_Full_Pre[k], 1)
                     ]...)[:, Impact_At_FFT[k]:(Impact_At_FFT[k] + Signal_Length_FFT - 1)]
                for k in eachindex(Traces_Full_Pre)
              ]
Phases_Full = [range(start=Phases_Angle[k][Times_Full[k][1]], stop=Phases_Angle[k][Times_Full[k][end]], length=FFT_Interpolation_Length)[Impact_At_FFT[k]:(Impact_At_FFT[k] + Signal_Length_FFT - 1)] for k in eachindex(Times_Full)]

JLSO.save("Data_Traces.bson", :Traces => Traces_Full_Cut, :Phases => Phases_Angle_Cut, :Time_Step => Time_Step)
end
