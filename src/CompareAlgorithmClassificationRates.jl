# ==========================================================================================
# COMPARISON OF CLASSIFICATION RATES BETWEEN 3 DIFFERENT LOCAL DISCRIMINANT BASES (LDB)
# ALGORITHMS
#
# This is the source code used to compare the differences in classification rates between 3
# LDB algorithms:
# ==========================================================================================

using Wavelets,
      WaveletsExt,
      Statistics,
      DataFrames,
      CSV,
      MLJ

include("mod/utils.jl")
import .LDBUtils: repeat_experiment, dict2dataframes, aggregate_results

## ========= Setup =========
# TODO: Change wavelet filter
wt = wavelet(WT.coif2)
# TODO: Change decomposition levels
L = 5
# TODO: Change the value of K
k = 10
# TODO: Change the number of features extracted
n_features = 10
# TODO： Change the type of discriminant power
dp = BasisDiscriminantMeasure()
# TODO: Change the discriminant measures
dm_ldbk    = AsymmetricRelativeEntropy()
dm_ldbkash = SymmetricRelativeEntropy()
dm_ldbkemd = EarthMoverDistance()
# Create LDB objects
ldbk    = LocalDiscriminantBasis(wt = wt, 
                                 max_dec_level = L,
                                 dm = dm_ldbk,
                                 en = TimeFrequency(),
                                 dp = dp,
                                 top_k = k,
                                 n_features = n_features)
ldbkash = LocalDiscriminantBasis(wt = wt, 
                                 max_dec_level = L,
                                 dm = dm_ldbkash,
                                 en = ProbabilityDensity(),
                                 dp = dp,
                                 top_k = k,
                                 n_features = n_features)
ldbkemd = LocalDiscriminantBasis(wt = wt, 
                                 max_dec_level = L,
                                 dm = dm_ldbkemd,
                                 en = Signatures(),
                                 dp = dp,
                                 top_k = k,
                                 n_features = n_features)
ldbs = Dict("original" => nothing, "ldbk" => ldbk, "ldbkash" => ldbkash, "ldbkemd" => ldbkemd)
# TODO: Make hyperparameter changes if necessary
LDA = @load LDA pkg=MultivariateStats verbosity=0
CT = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
classifiers = Dict("LDA" => LDA(), "CT" => CT())
# TODO: Make changes to model evaluation measures if necessary
measures = [MisclassificationRate(), MulticlassPrecision(), MulticlassTruePositiveRate()]

## ========== Run experiments ==========
# Set `save_data` to `true` to see the results in csv files.
results_raw = repeat_experiment(ldbs, classifiers, measures; repeats = 100, save_data=false)
results_by_measure = Dict("$((string ∘ typeof)(measure))" => dict2dataframes(results_raw, measure; save_data=false) for measure in measures)
results_aggregate = Dict(key => aggregate_results(value; save_data=false) for (key, value) in results_by_measure)
