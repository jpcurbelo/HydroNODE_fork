# --------------------------------------------------
# HydroNODE.jl
# Neural ODE models in Hydrology
# - load data
# - build models
# - train models
#
# marvin.hoege@eawag.ch, Aug. 2023 
# --------------------------------------------------

# Starting project from within
cd(@__DIR__)
## Uncomment this to run this module - it was commented out to run the parallel version HydroNODE_train_multiple.jl
# using Pkg; Pkg.activate(".."); Pkg.instantiate()

using Revise

using DataFrames, Dates, Statistics
using DelimitedFiles, CSV
using FilePathsBase

using OrdinaryDiffEq, DiffEqFlux, Lux
using ComponentArrays
using SciMLSensitivity

using Optimization, OptimizationOptimisers, OptimizationBBO
using Zygote

using Interpolations

# using Plots

using Random
Random.seed!(123)


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
basin_id = "01013500"

# define training and testing period
train_start_date = Date(1995,10,01)
train_stop_date = Date(2000,09,30)
test_start_date = Date(2000,10,01)
test_stop_date = Date(2005,09,30)

# train_start_date = Date(1980,10,01)
# train_stop_date = Date(2000,09,30)
# test_start_date = Date(2000,10,01)
# test_stop_date = Date(2010,09,30)

# if `false`, read the bucket model (M0) parameters from "bucket_opt_init.csv"
train_bucket_model = false

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
# Load and preprocess data

input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
output_var_name = "Flow(mm/s)"

df = load_data(lpad(string(basin_id), 8, "0"), data_path)

# drop unused cols
select!(df, Not(Symbol("SWE(mm)")));
select!(df, Not(Symbol("Tmax(C)")));
select!(df, Not(Symbol("Tmin(C)")));

# adjust start and stop date if necessary
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


# ===============================================================
# Bucket model training and full model preparation
NSE_loss_bucket_w_states(p) =  NSE_loss(basic_bucket_incl_states, p, train_y, train_timepoints)[1]

@info "Bucket model training..."

# Parameter ranges for bucket model:
# f: Rate of decline in flow from catchment bucket   | Range: (0, 0.1)
# smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
# qmax: Maximum subsurface flow at full bucket       | Range: (10, 50)
# ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
# tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
# tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)


if train_bucket_model == true

    callback_opt = function (p,l)
        println("NSE: "*string(-l))
        return false
    end

    lower_bounds = [0.01, 100.0, 0.0, 100.0, 10.0, 0.01, 0.0, -3.0]
    upper_bounds = [1500.0, 1500.0, 0.1, 1500.0, 50.0, 5.0, 3.0, 0.0]

    bounds_center = lower_bounds .+ 0.5*(upper_bounds.-lower_bounds)

    f = OptimizationFunction((θ, p) -> NSE_loss_bucket_w_states(θ))
    prob = Optimization.OptimizationProblem(f, bounds_center, lb = lower_bounds, ub = upper_bounds)
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = callback_opt, maxtime = 30.0)
    
    p_all_opt_bucket = sol.u

    S_bucket_precalib = p_all_opt_bucket[1:2]
    p_bucket_precalib = p_all_opt_bucket[3:8]
else

    bucket_opt_init = readdlm(joinpath(project_path,"param_set_files","bucket_opt_init.csv"), ',')
    basins_available = lpad.(string.(Int.(bucket_opt_init[:,1])), 8, "0")

    basin_wanted = findall(x -> x==basin_id, basins_available)[1]

    p_all_opt_bucket = bucket_opt_init[basin_wanted, 2:9]

    S_bucket_precalib = p_all_opt_bucket[1:2]
    p_bucket_precalib = p_all_opt_bucket[3:8]
end
@info "... complete!"

# println("Press Enter to continue...")
# readline()

# Solve the bucket model for the training data
Q_bucket, S_bucket  = basic_bucket_incl_states([S_bucket_precalib..., p_bucket_precalib...], train_timepoints)

# Select the dates from data_timepoints corresponding to train_timepoints indices
train_dates = collect(train_start_date:Day(1):train_stop_date)

# Prepare the data for saving with Date column corresponding to the selected dates
df_results = DataFrame(Date = train_dates, 
                       q_bucket = Q_bucket, 
                       q_obs = train_y,
                       s0 = S_bucket[1, :], 
                       s1 = S_bucket[2, :])

# Create a directory for the results if it doesn't exist - M0_results/basin_id
m0_results_path = joinpath(project_path, "M0_results")
if !isdir(m0_results_path)
    mkdir(m0_results_path)
end
results_path = joinpath(m0_results_path, basin_id)
if !isdir(m0_results_path)
    mkdir(m0_results_path)
end

# Save the DataFrame as a CSV file in the results directory
CSV.write(joinpath(m0_results_path, "bucket_model_results.csv"), df_results)

# Calculate Nash-Sutcliffe-Efficiency for the bucket model
NSE_opt_bucket = -NSE_loss_bucket_w_states(p_all_opt_bucket)
@info "NSE bucket model (train): $NSE_opt_bucket"

# # Plot and save bucket model results for training data
# scatter(train_timepoints, train_y, markersize = 2, markercolor = :black, label = "Training Data")
# plot!(train_timepoints, Q_bucket, color = :blue, label = "M0")

# # Create a directory for the plots if it doesn't exist
# plot_path = joinpath(project_path, "plots")
# if !isdir(plot_path)
#     mkdir(plot_path)
# end

# # Save the training data plot to a file in the plots directory
# savefig(joinpath(plot_path, "bucket_model_train.png"))

# # Plot and save bucket model results for testing data
# scatter(test_timepoints, test_y, markersize = 2, markercolor = :black, label = "Testing Data")
# plot!(test_timepoints, Q_NODE_test, color = :red, label = "NeuralODE Test")

# # Save the testing data plot to a file in the plots directory
# savefig(joinpath(plot_path, "bucket_model_test.png"))


# ===============================================================
# Neural ODE models

# -------------
# preparation

norm_S0 = prep_norm([mean(S_bucket[1,:]), std(S_bucket[1,:])])
norm_S1 = prep_norm([mean(S_bucket[2,:]), std(S_bucket[2,:])])

NN_NODE, p_NN_init = initialize_NN_model(chosen_model_id)

S0_bucket_   = S_bucket[1,:]
S1_bucket_   = S_bucket[2,:]
Lday_bucket_ = train_x[:,1]
P_bucket_    = train_x[:,2]
T_bucket_    = train_x[:,3]

NN_input = [norm_S0.(S0_bucket_) norm_S1.(S1_bucket_) norm_P.(P_bucket_) norm_T.(T_bucket_)]

@info "NN pre-training..."
p_NN_init = pretrain_NNs_for_bucket_processes(chosen_model_id, NN_NODE, p_NN_init,
    NN_input, p_bucket_precalib, S0_bucket_, S1_bucket_, Lday_bucket_, P_bucket_, T_bucket_)
@info "... complete!"

pred_NODE_model= prep_pred_NODE(NN_NODE, p_bucket_precalib[6:-1:4], S_bucket_precalib, length.(p_NN_init)[1])

NSE_init_NODE = -NSE_loss(pred_NODE_model,p_NN_init, train_y, train_timepoints)[1]
@info "NSE NODE init: $NSE_init_NODE"

# -------------
# training

@info "Neural ODE model training..."
p_opt_NODE = train_model(pred_NODE_model, p_NN_init, train_y, train_timepoints; optmzr = ADAM(0.0001), max_N_iter = 5)
@info "... complete."

# Q_NODE = pred_NODE_model(p_opt_NODE, train_timepoints)
Q_NODE_train = pred_NODE_model(p_opt_NODE, train_timepoints)
Q_sim_train = Q_NODE_train[1]
Q_NODE_test = pred_NODE_model(p_opt_NODE, test_timepoints)
Q_sim_test = Q_NODE_test[1]

# # # Print size of Q_NODE
# # @info "Q_NODE size: $(size(Q_NODE))"
# @info "Q_sim_train size: $(size(Q_sim_train[1]))"
# @info "ODESolution size: $(size(Q_sim_train[2].u))"
# # @info "Q_sim_test size: $(size(Q_sim_test))"




# # # # NSE_opt_NODE = -NSE_loss(pred_NODE_model,p_opt_NODE, train_y, train_timepoints)[1]
# # # # function NSE_loss(pred_model, params, batch, time_batch)

# # # #     pred, = pred_model(params, time_batch)

# Create a directory for the results if it doesn't exist - M100_results/basin_id
m100_results_path = joinpath(project_path, "M100_results")
if !isdir(m100_results_path)
    mkdir(m100_results_path)
end
results_path = joinpath(m0_results_path, basin_id)
if !isdir(m100_results_path)
    mkdir(m100_results_path)
end


# Save the model results for the Neural ODE model - training data
df_results_train = DataFrame(Date = train_dates, 
                    y_obs = train_y,
                    y_sim = Q_sim_train)

# Save the DataFrame as a CSV file in the results directory
CSV.write(joinpath(m100_results_path, "hybrid_model_results_train.csv"), df_results_train)

# Save the model results for the Neural ODE model - testing data
test_dates = collect(test_start_date:Day(1):test_stop_date)
df_results_test = DataFrame(Date = test_dates, 
                    y_obs = test_y,
                    y_sim = Q_sim_test)

# Save the DataFrame as a CSV file in the results directory
CSV.write(joinpath(m100_results_path, "hybrid_model_results_test.csv"), df_results_test)                      

@info "Results saved in $results_path"


# # plot NeuralODE results
# plot!(train_timepoints, Q_NODE, color = :green, label = chosen_model_id)

# -------------
# comparison bucket vs. Neural ODE

NSE_opt_NODE = -NSE_loss(pred_NODE_model,p_opt_NODE, train_y, train_timepoints)[1]
NSE_opt_NODE_test = -NSE_loss(pred_NODE_model,p_opt_NODE, test_y, test_timepoints)[1]

@info "Nash-Sutcliffe-Efficiency comparison (optimal value: 1):"

@info "NSE bucket model (train): $NSE_opt_bucket"

@info "NSE NeuralODE model (train): $NSE_opt_NODE"
@info "NSE NeuralODE model (test) : $NSE_opt_NODE_test"







