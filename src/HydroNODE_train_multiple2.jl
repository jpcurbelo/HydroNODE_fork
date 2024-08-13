# --------------------------------------------------
# Jesus Perez Curbelo - Aug. 2024
# --------------------------------------------------


# Starting project from within
cd(@__DIR__)
using Pkg; Pkg.activate(".."); Pkg.instantiate()

# @time using Revise

# @time using DataFrames, Dates, Statistics
# @time using DelimitedFiles, CSV

# @time using OrdinaryDiffEq, DiffEqFlux, Lux
@time using OrdinaryDiffEq
@time using DiffEqFlux
@time using Lux
# @time using ComponentArrays
# @time using SciMLSensitivity

# @time using Optimization, OptimizationOptimisers, OptimizationBBO
# @time using Zygote

# @time using Interpolations

# @time using Plots

# @time using Random
# @time Random.seed!(123)

@info "Loading packages complete"