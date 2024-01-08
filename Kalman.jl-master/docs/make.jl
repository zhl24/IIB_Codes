using Documenter
using Kalman

makedocs(
    modules = [Kalman],
    format = :html,
    sitename = "Kalman.jl",
    authors = "Moritz Schauer and contributors",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
#        "Manual" => "manual.md",
        "Library" => "library.md",
        ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mschauer/Kalman.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
