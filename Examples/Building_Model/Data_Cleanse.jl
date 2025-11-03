Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module dc
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels

Name = "Building_Model"
DATAVER = "V3"

Data_Dict = JLSO.load("Data_Traces.bson")
Time_Step = Data_Dict[:Time_Step]
Traces = Data_Dict[:Traces]
Phases = [ones(1, size(tr, 2)) for tr in Traces]

Signals, Phases_Embed = Delay_Embed(Traces, Phases; delay = 32, skip = 1)

Testing_Indices = [3;5;7]
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
)

end
