Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")

module dc
using LinearAlgebra
using Tullio
using JLSO
using Random
using Main.InvariantModels

Name = "Foil_Forced"
# DATAVER = "V3-1"
# PCA_Indices = 1:2
DATAVER = "V4-29"
PCA_Indices = 1:3
Skew_Dimension = 15

Data_Dict = JLSO.load("Data_Traces.bson")
Time_Step = Data_Dict[:Time_Step]
Traces_Full_Pre = Data_Dict[:Traces]
Phases_Angle = Data_Dict[:Phases]
Phases = [Fourier_Interpolate(Fourier_Grid(Skew_Dimension), vec(ph)) for ph in Phases_Angle]

Transform = JLSO.load("../Foil-Auto/RAWDATA-Foil_Auto-V4-1.bson")[:Inverse_Transform]'
Traces_Full = [Transform * sig for sig in Traces_Full_Pre]

# XX = hcat(Traces_Full...)
# CC = XX * XX'
# F = svd(CC)
# Traces = [F.Vt[PCA_Indices, :] * sig for sig in Traces_Full]

Traces_Full_Delay, Phases_Delay = Delay_Embed(Traces_Full, Phases; delay = 29, skip = 1)

Testing_Indices = [1;5;10]
Training_Indices = setdiff(eachindex(Traces_Full_Delay), Testing_Indices)

List_of_Data     = Traces_Full_Delay[Training_Indices]
List_of_Phases   = Phases_Delay[Training_Indices]
List_of_Angles   = Phases_Angle[Training_Indices]
List_of_Data_T   = Traces_Full_Delay[Testing_Indices]
List_of_Phases_T = Phases_Delay[Testing_Indices]
List_of_Angles_T = Phases_Angle[Testing_Indices]

JLSO.save(
    "RAWDATA-$(Name)-$(DATAVER).bson",
    :Time_Step => Time_Step,
    :List_of_Data => List_of_Data,
    :List_of_Phases => List_of_Phases,
    :List_of_Data_T => List_of_Data_T,
    :List_of_Phases_T => List_of_Phases_T,
    :List_of_Angles => List_of_Angles,
    :List_of_Angles_T => List_of_Angles_T,
    :Inverse_Transform => Transform',
)

end
