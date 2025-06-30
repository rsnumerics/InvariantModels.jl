using InvariantModels
using Documenter

DocMeta.setdocmeta!(InvariantModels, :DocTestSetup, :(using InvariantModels); recursive=true)

makedocs(;
    modules=[InvariantModels],
    authors="Robert Szalai <r.szalai@bristol.ac.uk> and contributors",
    sitename="InvariantModels.jl",
    format=Documenter.HTML(;
        canonical="https://rsnumerics.github.io/InvariantModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rsnumerics/InvariantModels.jl",
    devbranch="main",
)
