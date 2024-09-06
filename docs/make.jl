using ShaneGPUCountMinSketch
using Documenter

DocMeta.setdocmeta!(ShaneGPUCountMinSketch, :DocTestSetup, :(using ShaneGPUCountMinSketch); recursive=true)

makedocs(;
    modules=[ShaneGPUCountMinSketch],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="ShaneGPUCountMinSketch.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/ShaneGPUCountMinSketch.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/ShaneGPUCountMinSketch.jl",
    devbranch="main",
)
