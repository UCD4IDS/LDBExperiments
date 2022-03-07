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

include("utils.jl")

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
ldbs = Dict("ldbk" => ldbk, "ldbkash" => ldbkash, "ldbkemd" => ldbkemd)
# TODO: Make hyperparameter changes if necessary
LDA = @load LDA pkg=MultivariateStats
CT = @load DecisionTreeClassifier pkg=DecisionTrees
classifiers = Dict("LDA" => LDA(), "CT" => CT())
# TODO: Make changes to model evaluation measures if necessary
measures = [MisclassificationRate()]

## ========== Run experiments ==========
results = repeat_experiment(ldbs, classifiers, measurs)