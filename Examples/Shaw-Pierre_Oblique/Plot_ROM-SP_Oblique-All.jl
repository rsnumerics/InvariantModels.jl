
Code_Root = "../../src/"
include(Code_Root * "InvariantModels.jl")
module ct
using Main.InvariantModels
# using LinearAlgebra
# using Tullio
using JLSO
# using Random
#
using GLMakie, CairoMakie
using FFTW

fig_Auto = Figure(size = (1250, 750), fontsize = 32)
fig_Forced = Figure(size = (1250, 750), fontsize = 32)
Modes = 2
axes_Auto_All = [Axis(fig_Auto[j,k]) for j in 1:1+Modes, k in 1:Modes]
axes_Forced_All = [Axis(fig_Forced[j,k]) for j in 1:1+Modes, k in 1:Modes]
Segment_Start = 1
Segment_End = 5

# labels_Top_Auto = []
# labels_Side_Auto = []
# labels_Top_Forced = []
# labels_Side_Forced = []
# for k in 1:Modes
#     push!(labels_Top_Auto, Label(fig_Auto[0, k], "Foliation $(k)", tellheight=false, tellwidth=false))
#     push!(labels_Top_Forced, Label(fig_Forced[0, k], "Foliation $(k)", tellheight=false, tellwidth=false))
# end
# push!(labels_Side_Auto, Label(fig_Auto[1, 0], "Latent [pixels]", rotation = pi/2, tellheight=false, tellwidth=false))
# push!(labels_Side_Forced, Label(fig_Forced[1, 0], "Latent [pixels]", rotation = pi/2, tellheight=false, tellwidth=false))
# for k in 1:Modes
#     push!(labels_Side_Auto, Label(fig_Auto[1+k, 0], "Mode $(k) [pixels]", rotation = pi/2, tellheight=false, tellwidth=false))
#     push!(labels_Side_Forced, Label(fig_Forced[1+k, 0], "Mode $(k) [pixels]", rotation = pi/2, tellheight=false, tellwidth=false))
# end
for k in 1:Modes
    axes_Auto_All[1,k].title = "Autonomous Foliation $(k)"
    axes_Forced_All[1,k].title = "Forced Foliation $(k)"
end
axes_Auto_All[1,1].ylabel = "Latent"
axes_Forced_All[1,1].ylabel = "Latent"
for k in 1:Modes
    axes_Auto_All[1+k,1].ylabel = L"x_$(k)"
    axes_Forced_All[1+k,1].ylabel = L"x_$(k)"
end

All_Labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)", "o)", "p)", "q)", "r)"]
Name = "SP_Oblique"

let Label_Id= 1
    for Index in 1:Modes

        VER = "Autonomous"

        curves = JLSO.load("CURVES-$(Name)-$(VER)-I$(Index).bson")
        ODE_Backbone = curves[:ODE_Backbone]
        MAP_Backbone = curves[:MAP_Backbone]
        DATA_Backbone = curves[:DATA_Backbone]
        DATA_Error_Curves = curves[:DATA_Error_Curves]
        TEST_Error_Curves = curves[:TEST_Error_Curves]
        Error_Trace = curves[:Error_Trace]
        Test_Trace = curves[:Test_Trace]

        Data_Max = maximum(DATA_Error_Curves.Density_Amplitude)
        fig = Create_Plot()
        Plot_Backbone_Curves!(fig, MAP_Backbone, Data_Max; Label = "Map aut", Color = Makie.wong_colors()[2])
        Plot_Backbone_Curves!(fig, ODE_Backbone, Data_Max; Label = "ODE aut", Color = Makie.wong_colors()[3])
        Plot_Backbone_Curves!(fig, DATA_Backbone, Data_Max; Label = "Data aut", Color = Makie.wong_colors()[1])
        Plot_Error_Curves!(fig, DATA_Error_Curves, Data_Max; Label = "Data aut", Color = Makie.wong_colors()[1])
        Plot_Error_Curves!(fig, TEST_Error_Curves, Data_Max; Label = "Data aut", Color = Makie.wong_colors()[1],
                        Dense_Style = :dash,  Max_Style = :dashdot, Mean_Style = :dash)
        Plot_Error_Trace(fig, Index, Error_Trace, Test_Trace; Train_Color = Makie.wong_colors()[1], Test_Color = Makie.wong_colors()[1])
#         if Index == 2
#             axFreq = content(fig[1, 4])
#             axDamp = content(fig[1, 5])
#             xlims!(axFreq, 33.0, 33.5)
#             xlims!(axDamp, 0, 0.06)
#             axDense = content(fig[1, 1])
#             axErr = content(fig[1, 2])
#             ylims!(axFreq, 0, 6)
#             ylims!(axDamp, 0, 6)
#             ylims!(axDense, 0, 6)
#             ylims!(axErr, 0, 6)
#         end
        # Annotate_Plot!(fig)

        X_Axis = range(0, step = curves[:Time_Step], length = curves[:Index_List][Segment_End] - curves[:Index_List][Segment_Start])
        lines!(axes_Auto_All[1, Index], X_Axis, curves[:Latent_Data][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[1])
        lines!(axes_Auto_All[1, Index], X_Axis, curves[:Latent_Model][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[2])
        lines!(axes_Auto_All[1, Index], X_Axis, curves[:Latent_Data][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]] - curves[:Latent_Model][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[3])
        text!(axes_Auto_All[1, Index], All_Labels[Label_Id], space = :relative, position = Point2f(0.9, 0.7))
        Label_Id += 1
        for k in 1:Modes
            lines!(axes_Auto_All[1+k, Index], X_Axis, curves[:Data_Raw][k, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[1])
            lines!(axes_Auto_All[1+k, Index], X_Axis, curves[:Data_Reconstructed][k, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[2])
            text!(axes_Auto_All[1+k, Index], All_Labels[Label_Id], space = :relative, position = Point2f(0.9, 0.7))
            Label_Id += 1
        end
        axes_Auto_All[1+Modes, Index].xlabel = "Time [s]"

        VER = "Forced"

        curves = JLSO.load("CURVES-$(Name)-$(VER)-I$(Index).bson")
        ODE_Backbone = curves[:ODE_Backbone]
        MAP_Backbone = curves[:MAP_Backbone]
        DATA_Backbone = curves[:DATA_Backbone]
        DATA_Error_Curves = curves[:DATA_Error_Curves]
        TEST_Error_Curves = curves[:TEST_Error_Curves]
        Error_Trace = curves[:Error_Trace]
        Test_Trace = curves[:Test_Trace]

        Data_Max = maximum(DATA_Error_Curves.Density_Amplitude)
        Plot_Backbone_Curves!(fig, ODE_Backbone, Data_Max; Label = "ODE forced", Color = Makie.wong_colors()[6])
        Plot_Backbone_Curves!(fig, MAP_Backbone, Data_Max; Label = "Map forced", Color = Makie.wong_colors()[5])
        Plot_Backbone_Curves!(fig, DATA_Backbone, Data_Max; Label = "Data forced", Color = Makie.wong_colors()[4])
        Plot_Error_Curves!(fig, DATA_Error_Curves, Data_Max; Label = "Data forced", Color = Makie.wong_colors()[4])
        Plot_Error_Curves!(fig, TEST_Error_Curves, Data_Max; Label = "Data forced", Color = Makie.wong_colors()[4],
                        Dense_Style = :dash,  Max_Style = :dashdot, Mean_Style = :dash)
        Plot_Error_Trace(fig, Index, Error_Trace, Test_Trace; Train_Color = Makie.wong_colors()[4], Test_Color = Makie.wong_colors()[4])
        Annotate_Plot!(fig)

        CairoMakie.activate!(type = "svg")
        save("FIGURE-$(Name)-ALL-I$(Index).svg", fig)
        GLMakie.activate!()

        X_Axis = range(0, step = curves[:Time_Step], length = curves[:Index_List][Segment_End] - curves[:Index_List][Segment_Start])
        lines!(axes_Forced_All[1, Index], X_Axis, curves[:Latent_Data][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[1])
        lines!(axes_Forced_All[1, Index], X_Axis, curves[:Latent_Model][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[2])
        lines!(axes_Forced_All[1, Index], X_Axis, curves[:Latent_Data][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]] - curves[:Latent_Model][1, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[3])
        text!(axes_Forced_All[1, Index], All_Labels[Label_Id], space = :relative, position = Point2f(0.9, 0.7))
        Label_Id += 1
        for k in 1:Modes
            lines!(axes_Forced_All[1+k, Index], X_Axis, curves[:Data_Raw][k, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[1])
            lines!(axes_Forced_All[1+k, Index], X_Axis, curves[:Data_Reconstructed][k, 1+curves[:Index_List][Segment_Start]:curves[:Index_List][Segment_End]], color=Makie.wong_colors()[2])
            text!(axes_Forced_All[1+k, Index], All_Labels[Label_Id], space = :relative, position = Point2f(0.9, 0.7))
            Label_Id += 1
        end
        axes_Forced_All[1+Modes, Index].xlabel = "Time [s]"
    end
end

# fig_Shapes = Figure(size = (1000, 350), fontsize = 32)
# raw = JLSO.load("RAWDATA-Foil_Forced-V4-5.bson")
# ax_Shape = Axis(fig_Shapes[1,1])
# scatterlines!(ax_Shape, raw[:Inverse_Transform][:,1], label="Mode 1", linewidth=3, markersize=12)
# scatterlines!(ax_Shape, raw[:Inverse_Transform][:,2], label="Mode 2", linewidth=3, markersize=12)
# scatterlines!(ax_Shape, raw[:Inverse_Transform][:,3], label="Mode 3", linewidth=3, markersize=12)
# ax_Shape.xticks[] = 1:11
# ax_Shape.xlabel = "Pattern position (left to right)"
# ax_Shape.ylabel = "Weight"
# fig_Shapes[1, 2] = Legend(
#     fig_Shapes,
#     ax_Shape,
#     merge = true,
#     unique = true,
#     labelsize = 24,
#     backgroundcolor = (:white, 0),
#     framevisible = false,
#     rowgap = 1,
# )
# #
# fig_FFT = Figure(size = (1000, 350), fontsize = 32)
# ax_FFT = Axis(fig_FFT[1,1], yscale = log10)
# spec1 = abs.(fft(raw[:List_of_Data][5][1,:]))
# spec2 = abs.(fft(raw[:List_of_Data][5][2,:]))
# spec3 = abs.(fft(raw[:List_of_Data][5][3,:]))
# freq = range(0, 1 / raw[:Time_Step], length = length(spec1))
# lines!(ax_FFT, freq, spec1, label="Mode 1", linewidth=3)
# lines!(ax_FFT, freq, spec2, label="Mode 2", linewidth=3)
# lines!(ax_FFT, freq, spec3, label="Mode 3", linewidth=3)
# ax_FFT.xlabel = "Frequency [Hz]"
# ax_FFT.ylabel = "Amplitude [-]"
# xlims!(ax_FFT, (12, 40))
# ylims!(ax_FFT, (1, 1.2e4))
# fig_FFT[1, 2] = Legend(
#     fig_FFT,
#     ax_FFT,
#     merge = true,
#     unique = true,
#     labelsize = 24,
#     backgroundcolor = (:white, 0),
#     framevisible = false,
#     rowgap = 1,
# )

CairoMakie.activate!(type = "svg")
save("FIGURE-MODES-$(Name)-Autonomous.svg", fig_Auto)
save("FIGURE-MODES-$(Name)-Forced.svg", fig_Forced)
# save("FIGURE-MODE-SHAPES-All.svg", fig_Shapes)
# save("FIGURE-MODE-FFT.svg", fig_FFT)
GLMakie.activate!()

end
