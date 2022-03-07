function compute_ldb_vectors(ldb::LocalDiscriminantBasis, N::Integer)
    n = ldb.sz[1]                               # Signal length
    topN_subspace = ldb.order[1:N]              # Position of top N subspace
    vectors = zeros(n,N)                        # Build matrix to containing top N subspace
    for (j, i) in enumerate(topN_subspace)
        vectors[i,j] = 1
    end
    vectors = iwptall(vectors, wt, ldb.tree)    # Inverse transform matrix
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
                   X::AbstractArray{T₁}, y::AbstractVector{T₂}) where
                  {T₁<:AbstractFloat, T₂}
    # Get LDB features
    train_features = isnothing(ldb) ? X : WaveletsExt.transform(ldb, X)
    # Model fitting
    Xₜ = DataFrame(train_features', :auto)
    yₜ = coerce(y, Multiclass)
    model = machine(classifier, Xₜ, yₜ)
    MLJ.fit!(model)
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
                           repeats::Integer = 10, kwargs...) where
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
        results["experiment $i"] = resultᵢ
    end
    return results
end

h₁(i::Int) = max(6 - abs(i-7), 0)
h₂(i::Int) = h₁(i - 8)
h₃(i::Int) = h₁(i - 4)

"""
  triangular_test_function(c1::Int, c2::Int, c3::Int, L::Int=32)

Generates a set of triangluar test functions with 3 classes.
"""
function triangular_test_functions(c1::Int, c2::Int, c3::Int; L::Int=32, shuffle::Bool=false)
  @assert c1 >= 0
  @assert c2 >= 0
  @assert c3 >= 0

  u = rand(Uniform(0,1),1)[1]
  ϵ = rand(Normal(0,1),(L,c1+c2+c3))

  y = string.(vcat(ones(c1), ones(c2) .+ 1, ones(c3) .+ 2))

  H₁ = Array{Float64,2}(undef,L,c1)
  H₂ = Array{Float64,2}(undef,L,c2)
  H₃ = Array{Float64,2}(undef,L,c3)
  for i in 1:L
    H₁[i,:] .= u * h₁(i) + (1 - u) * h₂(i)
    H₂[i,:] .= u * h₁(i) + (1 - u) * h₃(i)
    H₃[i,:] .= u * h₂(i) + (1 - u) * h₃(i)
  end

  H = hcat(H₁, H₂, H₃) + ϵ

  if shuffle
    idx = [1:(c1+c2+c3)...]
    shuffle!(idx)
    return H[:,idx], y[idx]
  end

  return H, y
end