# ==========================================================================================
# COMPARISON OF FEATURE EXTRACTION BETWEEN 3 DIFFERENT LOCAL DISCRIMINANT BASIS (LDB) 
# ALGORITHMS
#
# This is the source code used to compare the differences in feature extraction capabilities
# between 3 LDB algorithms:
#   1) LDBK:
#           LDB that relies on the Time-Frequency energy distribution of classes. When
#           computing the discriminant measures of each node in the decomposed binary tree,
#           only the top K largest coefficients from each subspace are used.
#
#   2) LDBKASH:
#           LDB that relies on the Empirical Probability Densities of each class. When
#           computing the discriminant measures of each node in the decomposed binary tree,
#           only the top K largest coefficients from each subspace are used.
#
#   3) LDBKEMD:
#           LDB that depends on the use of signatures and Earth Mover Distance (EMD). When
#           computing the discriminant measures of each node in the decomposed binary tree,
#           only the top K largest coefficients from each subspace are used.
#
# In this file, the default values used to compare the algorithms are as follows:
#   1) LDBK:
#           Energy map          : Time-Frequency
#           Discriminant measure: Asymmetric Relative Entropy (Kullback-Leibler divergence)
#           K                   : 10
#           Discriminant power  : Output from discriminant measure
#
#   2) LDBKASH:
#           Energy map          : Empirical Probability Density
#           Discriminant measure: Asymmetric Relative Entropy (Kullback-Leibler divergence)
#           K                   : 10
#           Discriminant power  : Output from discriminant measure
#
#   3) LDBKEMD:
#           Energy map          : Signatures
#           Discriminant measure: Earth Mover's Distance
#           K                   : 10
#           Discriminant power  : Output from discriminant measure
#
#   Miscellaneous:
#           Dataset             : Triangular waveform dataset
#           Wavelet filter      : 6-tap Coiflet
#           Decomposition levels: 5
#           Train size          : 33/class
#           Test size           : 333/class
#           Features extracted  : 10
#
# The areas of the code that was set to the default values can be switched out. Look for the
# comments labeled with 
#           # TODO: ___________
# at the start.
#
# ------------------------------------------------------------------------------------------

using Wavelets, 
      WaveletsExt,
      Statistics,
      Plots,
      MLJ,
      DataFrames

include("utils.jl")
import .LDBUtils: compute_ldb_vectors, plot_coefficients

## ========== Setup ==========
train_x, train_y = generateclassdata(ClassData(:tri, 33 , 33 , 33 ), false)
test_x , test_y  = generateclassdata(ClassData(:tri, 333, 333, 333), true)
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

## ========== Display sample dataset ==========
# TODO: Change indexing if train size is changed above
class1_mean = mean(train_x[:, 1:33], dims=2)
class2_mean = mean(train_x[:,34:66], dims=2)
class3_mean = mean(train_x[:,67:99], dims=2)
p01 = plot(class1_mean, linestyle=:solid, color=:black, label="Class 1")
plot!(p01, class2_mean, linestyle=:dash, color=:black, label="Class 2")
plot!(p01, class3_mean, linestyle=:dot, color=:black, label="Class 3")
plot!(p01, title="Mean training waveform")

## ========== Fit LDB ==========
# Fit and transform training data
ldbk_train_features    = WaveletsExt.fit_transform(ldbk   , train_x, train_y)
ldbkash_train_features = WaveletsExt.fit_transform(ldbkash, train_x, train_y)
ldbkemd_train_features = WaveletsExt.fit_transform(ldbkemd, train_x, train_y)
# Transform test data
ldbk_test_features    = WaveletsExt.transform(ldbk, test_x)
ldbkash_test_features = WaveletsExt.transform(ldbkash, test_x)
ldbkemd_test_features = WaveletsExt.transform(ldbkemd, test_x)

## ========== Display top N LDB vectors ==========
# TODO: Change the value of N (Default: 3)
N = 3
ldbk_vec = compute_ldb_vectors(ldbk, N)
ldbkash_vec = compute_ldb_vectors(ldbkash, N)
ldbkemd_vec = compute_ldb_vectors(ldbkemd, N)
p02 = plot(ldbk_vec, label="")
p03 = plot(ldbkash_vec, label="")
p04 = plot(ldbkemd_vec, label="")

## ========== Display best basis nodes from each LDB algorithm ==========
p05 = plot_tfbdry(ldbk.tree)
p06 = plot_tfbdry(ldbkash.tree)
p07 = plot_tfbdry(ldbkemd.tree)

## ===== Scatter plot of coefficients in the top 2 most discriminating LDB coordinates =====
p08 = plot_coefficients(ldbk_train_features) |> p -> plot!(p, title="LDBK")
p09 = plot_coefficients(ldbkash_train_features) |> p -> plot!(p, title="LDBKASH")
p10 = plot_coefficients(ldbkemd_train_features) |> p -> plot!(p, title="LDBKEMD")

## ========== Fit features in Linear Discriminant Analysis (LDA) classifier ==========
# Data wrangling to fit MLJ.jl syntax
original_train_features = DataFrame(train_x', :auto)
ldbk_train_features     = DataFrame(ldbk_train_features', :auto)
ldbkash_train_features  = DataFrame(ldbkash_train_features', :auto)
ldbkemd_train_features  = DataFrame(ldbkemd_train_features', :auto)
train_y = coerce(train_y, Multiclass)
original_test_features = DataFrame(test_x', :auto)
ldbk_test_features     = DataFrame(ldbk_test_features', :auto)
ldbkash_test_features  = DataFrame(ldbkash_test_features', :auto)
ldbkemd_test_features  = DataFrame(ldbkemd_test_features', :auto)
test_y = coerce(test_y, Multiclass)
# Model fitting
LDA = @load LDA pkg=MultivariateStats
original_classifier = machine(LDA(), original_train_features, train_y)
ldbk_classifier     = machine(LDA(), ldbk_train_features, train_y)
ldbkash_classifier  = machine(LDA(), ldbkash_train_features, train_y)
ldbkemd_classifier  = machine(LDA(), ldbkemd_train_features, train_y)
MLJ.fit!(original_classifier)
MLJ.fit!(ldbk_classifier)
MLJ.fit!(ldbkash_classifier)
MLJ.fit!(ldbkemd_classifier)
# Model predictions
original_train_ŷ = predict_mode(original_classifier, original_train_features)
ldbk_train_ŷ     = predict_mode(ldbk_classifier    , ldbk_train_features)
ldbkash_train_ŷ  = predict_mode(ldbkash_classifier , ldbkash_train_features)
ldbkemd_train_ŷ  = predict_mode(ldbkemd_classifier , ldbkemd_train_features)
original_test_ŷ = predict_mode(original_classifier, original_test_features)
ldbk_test_ŷ     = predict_mode(ldbk_classifier    , ldbk_test_features)
ldbkash_test_ŷ  = predict_mode(ldbkash_classifier , ldbkash_test_features)
ldbkemd_test_ŷ  = predict_mode(ldbkemd_classifier , ldbkemd_test_features)
# Evaluate Model
original_train_accuracy = Accuracy()(original_train_ŷ, train_y)
@info "Original LDB Train Accuracy: $original_train_accuracy"
ldbk_train_accuracy     = Accuracy()(ldbk_train_ŷ    , train_y)
@info "LDBK Train Accuracy: $ldbk_train_accuracy"
ldbkash_train_accuracy  = Accuracy()(ldbkash_train_ŷ , train_y)
@info "LDBASH Train Accuracy: $ldbkash_train_accuracy"
ldbkemd_train_accuracy  = Accuracy()(ldbkemd_train_ŷ , train_y)
@info "LDBKEMD Train Accuracy: $ldbkemd_train_accuracy"
original_test_accuracy = Accuracy()(original_test_ŷ, test_y)
@info "Original LDB Test Accuracy: $original_test_accuracy"
ldbk_test_accuracy     = Accuracy()(ldbk_test_ŷ    , test_y)
@info "LDBK Test Accuracy: $ldbk_test_accuracy"
ldbkash_test_accuracy  = Accuracy()(ldbkash_test_ŷ , test_y)
@info "LDBKASH Test Accuracy: $ldbkash_test_accuracy"
ldbkemd_test_accuracy  = Accuracy()(ldbkemd_test_ŷ , test_y)
@info "LDBKEMD Test Accuracy: $ldbkemd_test_accuracy"