using InvariantModels
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(InvariantModels, :DocTestSetup, :(using InvariantModels); recursive=true)
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)

makedocs(;
    modules=[InvariantModels],
    authors="Robert Szalai <r.szalai@bristol.ac.uk> and contributors",
    sitename="InvariantModels.jl",
    format=Documenter.HTML(;
        canonical="https://rsnumerics.github.io/InvariantModels.jl",
        edit_link="main",
        assets=String[],
        mathengine=Documenter.KaTeX(),
        size_threshold_warn=1_000_000,
        size_threshold=2_000_000,
        example_size_threshold=2_000_000,
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
#     root    = "..",
#     source  = ".",
#     build   = "docs/build",
    pages=[
        "Home" => "index.md",
        "Tutorial" => "Tutorial.md"
    ],
    expandfirst=["index.md"],
    pagesonly=true,
    draft=true,
    warnonly=true,
    plugins=[bib],
)

deploydocs(;
    repo="github.com/rsnumerics/InvariantModels.jl",
    devbranch="main",
)
