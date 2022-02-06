# ==========================================================================================
# COMPARISON OF 3 DIFFERENT LOCAL DISCRIMINANT BASIS (LDB) ALGORITHMS
#
# This is the source code used to compare the 3 LDB algorithms:
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
      Plots

## ========== Setup ==========
# TODO: Change dataset and data size
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
ldbk_features    = fit_transform(ldbk   , train_x, train_y)
ldbkash_features = fit_transform(ldbkash, train_x, train_y)
ldbkemd_features = fit_transform(ldbkemd, train_x, train_y)

## ========== Display top N LDB vectors ==========
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
function plot_coefficients(data)
    p = scatter(data[1,1:33], data[2,1:33], shape=:circle, color=:black, label="Class 1")
    scatter!(p, data[1,34:66], data[2,34:66], shape=:star5, color=:yellow, label="Class 2")
    scatter!(p, data[1,67:99], data[2,67:99], shape=:cross, color=:green, label="Class 3")
    return p
end

p08 = plot_coefficients(ldbk_features) |> p -> plot!(p, title="LDBK")
p09 = plot_coefficients(ldbkash_features) |> p -> plot!(p, title="LDBKASH")
p10 = plot_coefficients(ldbkemd_features) |> p -> plot!(p, title="LDBKEMD")

## ========== Fit features in Linear Discriminant Analysis (LDA) classifier ==========