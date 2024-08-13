# --------------------------------------------------
# Jesus Perez Curbelo - Aug. 2024
# --------------------------------------------------


# Starting project from within
cd(@__DIR__)
using Pkg; Pkg.activate(".."); Pkg.instantiate()

@time using Revise

@time using DataFrames, Dates, Statistics
@time using DelimitedFiles, CSV

@time using OrdinaryDiffEq, DiffEqFlux, Lux
@time using OrdinaryDiffEq
@time using DiffEqFlux
@time using Lux
@time using ComponentArrays
@time using SciMLSensitivity

@time using Optimization, OptimizationOptimisers, OptimizationBBO
@time using Zygote

@time using Interpolations

@time using Plots

@time using Random
@time Random.seed!(123)

@info "Loading packages complete"

# ===========================================================
# USER INPUT:

# set data directory
project_path = joinpath(pwd(), "..")
# data_path = joinpath(pwd(),"basin_dataset_public_v1p2")
data_path = joinpath(pwd(),"../../../../gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_US")
# data_path = joinpath(pwd(),"../../CAMELS_US")

# choose model M50 or M100 or full
chosen_model_id = "M100"

# choose basin id
# Load the basin list from ../param_set_files
basin_file_name = "569_basin_file_5f.txt"
basin_file = joinpath(project_path, "param_set_files", basin_file_name)
basin_list = readlines(basin_file)
println(basin_list)

# define training and testing period
# # To be globally defined in HydroNODE.jl
# define training and testing period
train_start_date = Date(1995,10,01)
train_stop_date = Date(2000,09,30)
test_start_date = Date(2000,10,01)
test_stop_date = Date(2005,09,30)
@info "Training period: $train_start_date to $train_stop_date"
@info "Testing period: $test_start_date to $test_stop_date"
# ===========================================================

includet("HydroNODE_data.jl")
includet("HydroNODE_models.jl")
includet("HydroNODE_training.jl")

# -------------------------------------------------------
# Objective function: Nash-Sutcliffe Efficiency

NSE(pred, obs) = 1 - sum((pred .- obs).^2) / sum((obs .- mean(obs)).^2)

function NSE_loss(pred_model, params, batch, time_batch)

    pred, = pred_model(params, time_batch)
    loss = -NSE(pred,batch)

    return loss, pred
end


# -------------------------------------------------------
# Loop over basins
for basin_id in basin_list

    println("Basin: ", basin_id)

    # -------------------------------------------------------
    # Load and preprocess data

    input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
    output_var_name = "Flow(mm/s)"

    df = load_data(lpad(string(basin_id), 8, "0"), data_path)

    # drop unused cols
    select!(df, Not(Symbol("SWE(mm)")));
    select!(df, Not(Symbol("Tmax(C)")));
    select!(df, Not(Symbol("Tmin(C)")));

    # adjust start and stop date if necessary
    global train_start_date, test_stop_date
    if df[1, "Date"] != train_start_date
        train_start_date = maximum([df[1, "Date"],train_start_date])
    end

    if df[end, "Date"] != test_stop_date
        test_stop_date = minimum([df[end, "Date"], test_stop_date])
    end

    # Preparing Training and Testing Data
    # format data
    data_x, data_y, data_timepoints,
    train_x, train_y, train_timepoints, 
    test_x, test_y, test_timepoints = prepare_data(df,
    (train_start_date, train_stop_date, test_start_date, test_stop_date),input_var_names,output_var_name)

    # normalize data
    norm_moments_in = [mean(data_x, dims=1); std(data_x, dims = 1)]

    norm_P = prep_norm(norm_moments_in[:,2])
    norm_T = prep_norm(norm_moments_in[:,3])

    # -------------------------------------------------------
    # interpolation

    itp_method = SteffenMonotonicInterpolation()

    itp_Lday = interpolate(data_timepoints, data_x[:,1], itp_method)
    itp_P = interpolate(data_timepoints, data_x[:,2], itp_method)
    itp_T = interpolate(data_timepoints, data_x[:,3], itp_method)

end


