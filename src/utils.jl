module LDBUtils

export compute_ldb_vectors,
       plot_coefficients,
       extract_feature!,
       fit_model,
       evaluate_model,
       run_experiment,
       repeat_experiment,
       aggregate_results

using Wavelets,
      WaveletsExt,
      Statistics,
      DataFrames,
      CSV,
      MLJ,
      Plots

import JLD2: save
import JSON

function compute_ldb_vectors(ldb::LocalDiscriminantBasis, N::Integer)
    n = ldb.sz[1]                               # Signal length
    topN_subspace = ldb.order[1:N]              # Position of top N subspace
    vectors = zeros(n,N)                        # Build matrix to containing top N subspace
    for (j, i) in enumerate(topN_subspace)
        vectors[i,j] = 1
    end
    vectors = iwptall(vectors, ldb.wt, ldb.tree)    # Inverse transform matrix
    return vectors
end

function plot_coefficients(data)
    p = scatter(data[1,1:33], data[2,1:33], shape=:circle, color=:black, label="Class 1")
    scatter!(p, data[1,34:66], data[2,34:66], shape=:star5, color=:yellow, label="Class 2")
    scatter!(p, data[1,67:99], data[2,67:99], shape=:cross, color=:green, label="Class 3")
    return p
end

"""

"""
function extract_feature!(ldb::Union{LocalDiscriminantBasis, Nothing},
                          X::AbstractArray{T₁}, y::AbstractVector{T₂};
                          return_plots::Bool = false,
                          N::Integer = 3) where 
                         {T₁<:AbstractFloat, T₂}
    # If no available LDB object, then plots shouldn't be returned
    @assert !(isnothing(ldb) && return_plots)
    # Fit LDB
    train_features = isnothing(ldb) ? X : WaveletsExt.fit_transform(ldb, X, y)
    # Build plots if necessary
    if return_plots
        ldbvec = compute_ldb_vectors(ldb, N)
        ldbvec_plot = plot(ldbvec, label="")
        tfbdry_plot = plot_tfbdry(ldb.tree)
        coef_plot = plot_coefficients(train_features)
        return (ldbvec_plot = ldbvec_plot, tfbdry_plot = tfbdry_plot, coef_plot = coef_plot)
    end
end

function fit_model(ldb::Union{LocalDiscriminantBasis, Nothing},
                   classifier::Supervised,
                   X::AbstractArray{T₁}, y::AbstractVector{T₂};
                   verbosity = 0) where
                  {T₁<:AbstractFloat, T₂}
    # Get LDB features
    train_features = isnothing(ldb) ? X : WaveletsExt.transform(ldb, X)
    # Model fitting
    Xₜ = DataFrame(train_features', :auto)
    yₜ = coerce(y, Multiclass)
    model = machine(classifier, Xₜ, yₜ)
    MLJ.fit!(model; verbosity=verbosity)
    return model
end

function evaluate_model(ldb::Union{LocalDiscriminantBasis, Nothing},
                        model::Machine,
                        X::AbstractArray{T₁}, y::AbstractVector{T₂},
                        measures::Vector{T₃}) where
                       {T₁<:AbstractFloat, T₂, T₃<:MLJ.MLJBase.Measure}
    # Get LDB features
    train_features = isnothing(ldb) ? X : WaveletsExt.transform(ldb, X)
    # Model prediction
    Xₜ = DataFrame(train_features', :auto)
    yₜ = coerce(y, Multiclass)
    ŷ = predict_mode(model, Xₜ)
    # Model evaluation
    result = Dict{String, Float64}()
    for measure in measures
        metric = typeof(measure)
        result["$metric"] = measure(ŷ, yₜ)
    end
    return result
end

function vector2dict(vec::Vector{T₁}, suffix::String = "method") where T₁
    return Dict("$(suffix)_$i" => data for (i, data) in enumerate(vec))
end

function run_experiment(ldbs::Union{Dict{String, T₁}, Vector{T₁}},
                        classifiers::Union{Dict{String, T₂}, Vector{T₂}},
                        X_train::AbstractArray{T₃}, y_train::AbstractVector{T₄}, 
                        X_test::AbstractArray{T₃}, y_test::AbstractVector{T₄}, 
                        measures::Vector{T₅}; kwargs...) where 
                       {T₁<:Union{LocalDiscriminantBasis, Nothing}, 
                        T₂<:Supervised,
                        T₃<:AbstractFloat, T₄,
                        T₅<:MLJ.MLJBase.Measure}
    ldbs = isa(ldbs, Vector) ? vector2dict(ldbs, "ldb") : ldbs
    classifiers = isa(classifiers, Vector) ? vector2dict(classifiers, "clf") : classifiers
    models = Dict(name => Dict{String, Machine}() for name in keys(ldbs))
    result = Dict(name => Dict{String, Dict}() for name in keys(ldbs))
    for (ldb_name, ldb) in ldbs
        for (clf_name, classifier) in classifiers
            # Extract LDB features
            extract_feature!(ldb, X_train, y_train)
            # Fit model
            models[ldb_name][clf_name] = fit_model(ldb, classifier, X_train, y_train)
            # Evaluate model
            train_result = evaluate_model(ldb, models[ldb_name][clf_name], X_train, y_train, measures)
            test_result = evaluate_model(ldb, models[ldb_name][clf_name], X_test, y_test, measures)
            result[ldb_name][clf_name] = Dict("train" => train_result, "test" => test_result)
        end
    end
    return (model = models, result = result)
end

function repeat_experiment(ldbs::Union{Dict{String, T₁}, Vector{T₁}},
                           classifiers::Union{Dict{String, T₂}, Vector{T₂}}, 
                           measures::Vector{T₃}; 
                           config_train::ClassData = ClassData(:cbf, 33, 33, 33),
                           config_test::ClassData = ClassData(:cbf, 333, 333, 333),
                           repeats::Integer = 10,
                           save_data::Bool = true,
                           kwargs...) where
                          {T₁<:Union{LocalDiscriminantBasis, Nothing}, 
                           T₂<:Supervised,
                           T₃<:MLJ.MLJBase.Measure}
    ldbs = isa(ldbs, Vector) ? vector2dict(ldbs, "ldb") : ldbs
    classifiers = isa(classifiers, Vector) ? vector2dict(classifiers, "clf") : classifiers
    results = Dict{String, Dict}()
    for i in 1:repeats
        X_train, y_train = generateclassdata(config_train, false)
        X_test, y_test = generateclassdata(config_test, true)
        _, resultᵢ = run_experiment(ldbs, classifiers, X_train, y_train, X_test, y_test, measures, kwargs...)
        results["$i"] = resultᵢ

        # Save data and results
        if save_data
            path = "./results/experiment_data"
            save("$path/exp$i.jld2",
                 "X_train", X_train,
                 "y_train", y_train,
                 "X_test", X_test,
                 "y_test", y_test)
            save_to_json("$path/result$i.json", resultᵢ)
        end
    end
    return results
end

function save_to_json(filepath::String, result::AbstractDict)
    open(filepath, "w") do f
        JSON.print(f, result, 4)
    end
end

"""
    dict2dataframes(results)

Organizes a dictionary of results into a list of data frames.

# Arguments
- `results::Dict`: Dictionary of results.

# Returns
- `Vector{DataFrame}`: List of data frames containing the organized results.
"""
function dict2dataframes(results::Dict, measure::T; kwargs...) where T<:MLJ.MLJBase.Measure
    measure_string = (string ∘ typeof)(measure)
    return dict2dataframes(results, measure_string; kwargs...)
end

function dict2dataframes(results::Dict, measure::String; save_data::Bool = true)
    data_frame = DataFrame(
        "Experiment" => UInt64[],
        "Method" => String[],
        "Classifier" => String[],
        "Train_$measure" => Float64[],
        "Test_$measure" => Float64[]
    )

    for (exp, exp_result) in results
        for (ldb, ldb_result) in exp_result
            for (clf, clf_result) in ldb_result
                data = [
                    parse(UInt64, exp),             # Experiment number
                    ldb,                            # LDB method
                    clf,                            # Classifier
                    clf_result["train"][measure],   # Train metric
                    clf_result["test"][measure]     # Test metric
                ]
                push!(data_frame, data)
            end
        end
    end

    if save_data
        CSV.write("./results/complete/$measure.csv", data_frame)
    end

    return data_frame
end


function get_measure_name(data::Vector{String})
    reg_match = [match(r"^(Train|Test)_([A-Za-z0-9]*)", str) for str in data]
    first_match_index = findall(!isnothing, reg_match)[1]
    return reg_match[first_match_index][2]
end

function aggregate_results(result::DataFrame; save_data::Bool = true)
    results_grouped = groupby(result, [:Method, :Classifier])
    results_combined = combine(results_grouped, r"^Train" => mean, r"^Test" => mean, renamecols = false)

    if save_data
        measure = (get_measure_name ∘ names)(result)
        CSV.write("./results/aggregate/$measure.csv", results_combined)
    end

    return results_combined
end

end # module