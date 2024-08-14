# --------------------------------------------------
# Jesus Perez Curbelo - Aug. 2024
# --------------------------------------------------


# Starting project from within
cd(@__DIR__)
using Pkg; Pkg.activate(".."); Pkg.instantiate()

# # Add the required package
# Pkg.add("FilePathsBase")

@time using Revise

@time using DataFrames, Dates, Statistics
@time using DelimitedFiles, CSV
@time using FilePathsBase

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

@time using Printf
# @time using Base.Threads

# # # Define a constant for the number of threads
# # const NUM_THREADS = 4  # or any number you desire

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
basin_file_name = "569_basin_file.txt"
# basin_file_name = "569_basin_file_16f.txt"
# basin_file_name = "569_basin_file_3f.txt"
# basin_file_name = "2_basin_file.txt"
basin_file = joinpath(project_path, "param_set_files", basin_file_name)
basin_list = readlines(basin_file)
# println(basin_list)

# Results directory names
m0_folder = "M0_results_569basins_1980_2010_BS3_julia"
# m0_folder = "M0_results_16basins"
# m0_folder = "M0_results_3basins"
# m0_folder = "M0_results_2basin"

m0_outputs_path = joinpath(project_path, m0_folder)
if !isdir(m0_outputs_path)
    mkdir(m0_outputs_path)
end
m0_results_path = joinpath(m0_outputs_path, "model_results")
if !isdir(m0_results_path)
    mkdir(m0_results_path)
end

# Define the maximum number of iterations and the callback frequency for the training 
MAX_N_ITER = 150
CALLBACK_FREQ = 10
LR_TRAIN = 0.0001  # Hoge value is 0.0001

m100_folder = "$(chosen_model_id)_results_569basins_1980_2010_BS3_julia_$(MAX_N_ITER)ep_lr4"
# m100_folder = "$(chosen_model_id)_results_16basins_$(MAX_N_ITER)ep_lr4"
# m100_folder = "$(chosen_model_id)_results_3basins"
# m100_folder = "$(chosen_model_id)_results_2basin_$(MAX_N_ITER)ep_lr4"

m100_outputs_path = joinpath(project_path, m100_folder)
if !isdir(m100_outputs_path)
    mkdir(m100_outputs_path)
end
m100_results_path = joinpath(m100_outputs_path, "model_results")
if !isdir(m100_results_path)
    mkdir(m100_results_path)
end

# define training and testing period
# train_start_date = Date(1995,10,01)
# train_stop_date = Date(2000,09,30)
# test_start_date = Date(2000,10,01)
# test_stop_date = Date(2005,09,30)

train_start_date = Date(1980,10,01)
train_stop_date = Date(2000,09,30)
test_start_date = Date(2000,10,01)
test_stop_date = Date(2010,09,30)

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

# Helper function to generate a versioned file name
function get_versioned_filename(filepath::AbstractString)
    @info "Generating versioned file name for $filepath"

    dir = dirname(filepath)
    # ext = Base.extname(filepath)  # Corrected to Base.extname
    ext = split(filepath, ".")[end]
    # base = basename(filepath, ext)
    base = basename(filepath)
    # Drop the extension from the base name
    base = split(base, ".")[1]

    # Find the numerical suffix (e.g., "batch1") and split it from the rest of the name
    m = match(r"^(.*_batch)(\d+)$", base)
    if m !== nothing
        base_part, number_part = m.captures
        number = parse(Int, number_part)
    else
        # If no suffix exists, start at batch1
        base_part = base
        number = 1
    end

    # Generate versioned file name
    while true
        versioned_filename = joinpath(dir, "$(base_part)$(number)$(ext)")
        if isfile(versioned_filename)
            number += 1
        else
            filepath = versioned_filename
            break
        end
    end

    return filepath
end


# Struct to store basin metrics
struct BasinMetrics
    basin_id::String
    NSE_train::Float32
    NSE_test::Float32
end
basin_metrics_list_m0 = []
basin_metrics_list_pretrainer = []
basin_metrics_list_m100 = []
##################################################

USING_BATCH = true
batch1 = basin_list[1:70]
batch2 = basin_list[71:140]
batch3 = basin_list[141:210]
batch4 = basin_list[211:280]
batch5 = basin_list[281:350]
batch6 = basin_list[351:420]
batch7 = basin_list[421:490]
batch8 = basin_list[491:end]  # Or basin_list[491:569]
# -------------------------------------------------------
# Loop over basins
# for basin_id in basin_list
for basin_id in batch8

    println("="^80)
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

    global norm_P, norm_T
    norm_P = prep_norm(norm_moments_in[:,2])
    norm_T = prep_norm(norm_moments_in[:,3])

    # -------------------------------------------------------
    # interpolation

    itp_method = SteffenMonotonicInterpolation()

    global itp_Lday, itp_P, itp_T
    itp_Lday = interpolate(data_timepoints, data_x[:,1], itp_method)
    itp_P = interpolate(data_timepoints, data_x[:,2], itp_method)
    itp_T = interpolate(data_timepoints, data_x[:,3], itp_method)

    # ===============================================================
    # Bucket model training and full model preparation
    NSE_loss_bucket_w_states(p) =  NSE_loss(basic_bucket_incl_states, p, train_y, train_timepoints)[1]

    NSE_loss_bucket_w_states_test(p) =  NSE_loss(basic_bucket_incl_states, p, test_y, test_timepoints)[1]

    @info "Bucket model parameters - loading..."
    
    bucket_opt_init = readdlm(joinpath(project_path,"param_set_files","bucket_opt_init.csv"), ',')
    basins_available = lpad.(string.(Int.(bucket_opt_init[:,1])), 8, "0")

    basin_wanted = findall(x -> x==basin_id, basins_available)[1]

    p_all_opt_bucket = bucket_opt_init[basin_wanted, 2:9]

    S_bucket_precalib = p_all_opt_bucket[1:2]
    p_bucket_precalib = p_all_opt_bucket[3:8]

    @info "Solving the bucket model for the training and testing data..."
    start_time_concept = time_ns()

    # Solve the bucket model for the training data
    Q_bucket, S_bucket  = basic_bucket_incl_states([S_bucket_precalib..., p_bucket_precalib...], train_timepoints)

    # Shift the S_bucket initial conditions to the end of the train time series
    S_bucket_precalib_test = [S_bucket[1, end], S_bucket[2, end]]

    # Create  p_all_opt_bucket_test and update the initial conditions
    p_all_opt_bucket_test = [S_bucket_precalib_test..., p_bucket_precalib...]

    # Solve the bucket model for the testing data
    Q_bucket_test, S_bucket_test  = basic_bucket_incl_states([S_bucket_precalib_test..., p_bucket_precalib...], test_timepoints)

    # Select the dates from data_timepoints corresponding to train_timepoints indices and test_timepoints indices
    train_dates = collect(train_start_date:Day(1):train_stop_date)
    test_dates = collect(test_start_date:Day(1):test_stop_date)

    # Prepare the data for saving with Date column corresponding to the selected dates
    df_results = DataFrame(Date = train_dates, 
                        q_bucket = Q_bucket, 
                        q_obs = train_y,
                        s0 = S_bucket[1, :], 
                        s1 = S_bucket[2, :])

    # Save the DataFrame as a CSV file in the results directory
    results_file = joinpath(m0_results_path, "$(basin_id)_results_train.csv")

    CSV.write(results_file, df_results)

    df_results_test = DataFrame(Date = test_dates, 
                        q_bucket = Q_bucket_test, 
                        q_obs = test_y,
                        s0 = S_bucket_test[1, :], 
                        s1 = S_bucket_test[2, :])

    # Save the DataFrame as a CSV file in the results directory
    results_file_test = joinpath(m0_results_path, "$(basin_id)_results_test.csv")

    CSV.write(results_file_test, df_results_test)

    # Calculate Nash-Sutcliffe-Efficiency for the bucket model
    NSE_opt_bucket = -NSE_loss_bucket_w_states(p_all_opt_bucket)
    @info "NSE bucket model (train): $(Printf.@sprintf("%.5f", NSE_opt_bucket))"

    NSE_opt_bucket_test = -NSE_loss_bucket_w_states_test(p_all_opt_bucket_test)
    @info "NSE bucket model (test) : $(Printf.@sprintf("%.5f", NSE_opt_bucket_test))"

    # Store the basin metrics in the basin_metrics_list_m0
    push!(basin_metrics_list_m0, BasinMetrics(basin_id, NSE_opt_bucket, NSE_opt_bucket_test))

    elapsed_time_ns_concept = time_ns() - start_time_concept
    elapsed_time_s_concept = elapsed_time_ns_concept / 1e9  # Convert nanoseconds to seconds
    @info "... completed in $(Printf.@sprintf("%.2f", elapsed_time_s_concept)) s"

    # ===============================================================
    # Neural ODE models

    global norm_S0, norm_S1
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
    start_time_pretrain = time_ns()

    p_NN_init = pretrain_NNs_for_bucket_processes(chosen_model_id, NN_NODE, p_NN_init,
        NN_input, p_bucket_precalib, S0_bucket_, S1_bucket_, Lday_bucket_, P_bucket_, T_bucket_)

    elapsed_time_ns_pretrain =  time_ns() - start_time_pretrain
    elapsed_time_s_pretrain = elapsed_time_ns_pretrain / 1e9  # Convert nanoseconds to seconds
    @info "... completed in $(Printf.@sprintf("%.2f", elapsed_time_s_pretrain)) s"

    # Train set
    pred_NODE_model= prep_pred_NODE(NN_NODE, p_bucket_precalib[6:-1:4], S_bucket_precalib, length.(p_NN_init)[1])

    NSE_init_NODE = -NSE_loss(pred_NODE_model,p_NN_init, train_y, train_timepoints)[1]
    @info "NSE NODE init (train): $(Printf.@sprintf("%.5f", NSE_init_NODE))"

    # Test set
    pred_NODE_model_test= prep_pred_NODE(NN_NODE, p_bucket_precalib[6:-1:4], S_bucket_precalib_test, length.(p_NN_init)[1])

    NSE_init_NODE_test = -NSE_loss(pred_NODE_model_test,p_NN_init, test_y, test_timepoints)[1]
    @info "NSE NODE init (test) : $(Printf.@sprintf("%.5f", NSE_init_NODE_test))"

    # Store the basin metrics in the basin_metrics_list_pretrainer
    push!(basin_metrics_list_pretrainer, BasinMetrics(basin_id, NSE_init_NODE, NSE_init_NODE_test))

    # -------------
    # training

    @info "Neural ODE model training..."
    start_time_train = time_ns()

    p_opt_NODE = train_model(pred_NODE_model, p_NN_init, train_y, train_timepoints; optmzr = ADAM(LR_TRAIN), max_N_iter = MAX_N_ITER, callback_freq = CALLBACK_FREQ)
    
    elapsed_time_ns_train = time_ns() - start_time_train
    elapsed_time_s_train = elapsed_time_ns_train / 1e9  # Convert nanoseconds to seconds
    @info "... completed in $(Printf.@sprintf("%.2f", elapsed_time_s_train)) s"


    Q_NODE_train = pred_NODE_model(p_opt_NODE, train_timepoints)
    Q_sim_train = Q_NODE_train[1]
    Q_NODE_test = pred_NODE_model(p_opt_NODE, test_timepoints)
    Q_sim_test = Q_NODE_test[1]

    # Save the model results for the Neural ODE model - training data
    df_results_train = DataFrame(Date = train_dates, 
                        y_obs = train_y,
                        y_sim = Q_sim_train)

    # Save the DataFrame as a CSV file in the results directory
    results_file_train = joinpath(m100_results_path, "$(basin_id)_results_train.csv")    
    
    CSV.write(results_file_train, df_results_train)

    # Save the model results for the Neural ODE model - testing data
    df_results_test = DataFrame(Date = test_dates, 
                        y_obs = test_y,
                        y_sim = Q_sim_test)

    # Save the DataFrame as a CSV file in the results directory
    results_file_test = joinpath(m100_results_path, "$(basin_id)_results_test.csv")

    CSV.write(results_file_test, df_results_test)

    @info "Results saved in $m100_results_path"

    # Store the basin metrics in the basin_metrics_list_m100
    NSE_NODE_train = -NSE_loss(pred_NODE_model, p_opt_NODE, train_y, train_timepoints)[1]
    NSE_NODE_test = -NSE_loss(pred_NODE_model, p_opt_NODE, test_y, test_timepoints)[1]

    push!(basin_metrics_list_m100, BasinMetrics(basin_id, NSE_NODE_train, NSE_NODE_test))

    @info "NSE NODE model (train): $(Printf.@sprintf("%.5f", NSE_NODE_train))"
    @info "NSE NODE model (test) : $(Printf.@sprintf("%.5f", NSE_NODE_test))"

    # ===============================================================

##################################################
end

# Convert the list of BasinMetrics to a DataFrame - M0
metrics_df_M0 = DataFrame(basin_id = [bm.basin_id for bm in basin_metrics_list_m0],
                       NSE_train = [bm.NSE_train for bm in basin_metrics_list_m0],
                       NSE_test = [bm.NSE_test for bm in basin_metrics_list_m0])

# # Save the DataFrame as a CSV file in the results directory
if USING_BATCH
    metrics_file_M0 = get_versioned_filename(joinpath(m0_outputs_path, "model_metrics_batch1.csv"))
else
    metrics_file_M0 = joinpath(m0_outputs_path, "model_metrics.csv")
end
CSV.write(metrics_file_M0, metrics_df_M0)

# Convert the list of BasinMetrics to a DataFrame - Pretrainer
metrics_df_pretrainer = DataFrame(basin_id = [bm.basin_id for bm in basin_metrics_list_pretrainer],
                       NSE_train = [bm.NSE_train for bm in basin_metrics_list_pretrainer],
                       NSE_test = [bm.NSE_test for bm in basin_metrics_list_pretrainer])

# Save the DataFrame as a CSV file in the results directory
if USING_BATCH
    metrics_file_pretrainer = get_versioned_filename(joinpath(m100_outputs_path, "model_metrics_pretrainer_batch1.csv"))
else
    metrics_file_pretrainer = joinpath(m100_outputs_path, "model_metrics_pretrainer.csv")
end
CSV.write(metrics_file_pretrainer, metrics_df_pretrainer)

# Convert the list of BasinMetrics to a DataFrame - M100
metrics_df_M100 = DataFrame(basin_id = [bm.basin_id for bm in basin_metrics_list_m100],
                       NSE_train = [bm.NSE_train for bm in basin_metrics_list_m100],
                       NSE_test = [bm.NSE_test for bm in basin_metrics_list_m100])

# Save the DataFrame as a CSV file in the results directory
if USING_BATCH
    metrics_file_M100 = get_versioned_filename(joinpath(m100_outputs_path, "model_metrics_node_batch1.csv"))
else
    metrics_file_M100 = joinpath(m100_outputs_path, "model_metrics_node.csv")
end
CSV.write(metrics_file_M100, metrics_df_M100)