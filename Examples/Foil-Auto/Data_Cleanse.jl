Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module dc
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels

Name = "Foil_Auto"
# DATAVER = "V3-1"
# PCA_Indices = 1:2
DATAVER = "V4-1"
PCA_Indices = 1:3

Data_Dict = JLSO.load("Data_Traces.bson")
Time_Step = Data_Dict[:Time_Step]
Traces_Full = Data_Dict[:Traces]
Phases = [ones(1, size(tr, 2)) for tr in Traces_Full]

XX = hcat(Traces_Full...)
CC = XX * XX'
F = svd(CC)
Traces = [F.Vt[PCA_Indices, :] * sig for sig in Traces_Full]

Signals, Phases_Embed = Delay_Embed(Traces, Phases; delay = 29, skip = 1)

Testing_Indices = [1;5;10]
Training_Indices = setdiff(eachindex(Signals), Testing_Indices)

List_of_Data     = Signals[Training_Indices]
List_of_Phases   = Phases_Embed[Training_Indices]
List_of_Data_T   = Signals[Testing_Indices]
List_of_Phases_T = Phases_Embed[Testing_Indices]

JLSO.save(
    "RAWDATA-$(Name)-$(DATAVER).bson",
    :Time_Step => Time_Step,
    :List_of_Data => List_of_Data,
    :List_of_Phases => List_of_Phases,
    :List_of_Data_T => List_of_Data_T,
    :List_of_Phases_T => List_of_Phases_T,
    :Inverse_Transform => F.U[:, PCA_Indices],
)

end
