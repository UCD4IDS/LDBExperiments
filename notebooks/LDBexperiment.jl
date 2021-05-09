### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 45468d3a-3456-4e99-aec8-b3c41b20a426
let
	import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 5ad9f0fb-3688-4b15-94c1-a18e5f41eeed
begin
	using 
		Random, 
		Statistics, 
		Distributions, 
		DataFrames,
		Wavelets, WaveletsExt,
		MLJ, 
		Gadfly,
		Plots,
		PlutoUI
end

# ╔═╡ f05ee9bc-3d1f-4ada-a4d7-0f44a5454646
include("../src/utils.jl")

# ╔═╡ 45f88030-a821-11eb-0c6d-f5c7c82b7115
md"# Local Discriminant Basis"

# ╔═╡ f3a892ef-0c5a-4d4a-86ab-036d913d9d97
md"## A Brief History"

# ╔═╡ c195f5d9-2538-4278-9d27-c14446e7cb65
md"**Local Discriminant Basis (LDB)** is a wavelet based feature extraction method concieved by Naoki Saito in 1992. Earlier that year, [Victor Wickerhauser](https://www.math.wustl.edu/~victor/) had generalized the best basis algorithm such that it worked not only for a single signal, but for a collection of signals that share the same important features. The so called Joint Best Basis (JBB) can be viewed as a time-frequency localized version of the Principle Component Analysis(PCA) or the Karhunen-Loève Basis (KLB).\
\
While JBB is good for signals belonging to the same class (i.e. share the same features), it does not work for signal classifications in general. LDB sets out to solve this issue by replacing the original minimum entropy cost function used in the JBB with the Kullback-Leiber divergence (a.k.a. relative entropy). More specifically,

1. Decomposes individual signals into time-frequency dictionaries. 
2. Creates a time-frequency energy distribution for each class by accumulating these dictionaries.
3. A complete orthonormal basis called LDB, which encodes the time-frequency localized similarity and differences between signal classes, is computed using distance measures such as KL-divergence or the Hellinger distance.
4. The coordinates that house the most distinguishing (the most divergent) feature of each signal class is identified.
5. The features at these coordinates are extracted from the time-frequency dictionaries of each individual signal and fed to classification algorithms such as Decision Trees (CART) of Linear Discriminant Analysis (LDA).\

For more on LDB, please visit the following resources:
* [Local Discriminant Basis](https://www.math.ucdavis.edu/~saito/contrib/#ldb)
* [On Local Orthonormal Bases for Classification and Regression](http://math.ucdavis.edu/~saito/publications/saito_icassp95.pdf)
* [Local discriminant bases and their applications](http://math.ucdavis.edu/~saito/publications/saito_ldb_jmiv.pdf)
"

# ╔═╡ a751cd87-80c5-48b1-b798-f1aecebc08a1
md"## A Simple Demonstration"

# ╔═╡ b8077eb3-ec64-4a84-9dcc-3aafce015597
md"We begin by manually generating some functions to classify. The `triangluar_test_functions` function will generate a set of test functions consisting of 3 classes."

# ╔═╡ dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
begin
	X₀, y₀ = triangular_test_functions(10,10,10);
	Y₀ = coerce(y₀, Multiclass); # For compatibility with MLJ.jl
end

# ╔═╡ a8f1e20b-e828-4c1b-80d0-999f5c3b6e5e
md"These test functions are generated using the following formulas:
* Class 1: $x^{(1)}(i) = uh_1(i) + (1-u)h_2(i)+\epsilon(i)$
* Class 2: $x^{(2)}(i) = uh_1(i) + (1-u)h_3(i)+\epsilon(i)$
* Class 3: $x^{(3)}(i) = uh_2(i) + (1-u)h_3(i)+\epsilon(i)$
where $i = 1,...,32$, $h_1(i)=max(6-|i-7|)$, $h_2(i)=h_1(i-8), h_3(i)=h_1(i-4)$,$u  \sim unif(0,1)$, and $\epsilon \sim N(0,1)$.
"

# ╔═╡ 59a3c5b3-d3c6-4b16-ae1b-984b6a77350a


# ╔═╡ 39f64d00-350d-43a6-bf57-06600aac2365
begin
	p1 = wiggle(X₀[:,1:5], sc=0.5)
	Plots.plot!(xlab = "Class 1")
	p2 = wiggle(X₀[:,11:15], sc=0.5)
	Plots.plot!(xlab = "Class 2")
	p3 = wiggle(X₀[:,21:25], sc=0.5)
	Plots.plot!(xlab = "Class 3")
	Plots.plot(p1, p2, p3, layout = (3,1))
end

# ╔═╡ 37a7c05f-eb0b-4509-9945-e482c4b9bc5a
md"We pass these functions and their labels to the `ldb` function in `WaveletsExt.jl`. By default, the `ldb` algorithm uses `Time Frequency` energy maps and the `Assymmetric Relative Entropy` (i.e. Kullback Leiber Divergence) to calculate obtain the LDB, but there are several other options which you can choose and experiment with."

# ╔═╡ 28604f68-a957-4a3c-92f5-13a0ff4ba158
@bind d_measure Radio(
	[
		"Asymmetric Relative Entropy",
		"Lp Entropy",
		"Symmetric Relative Entropy",
		"Hellinger Distance"
	],
	default = "Asymmetric Relative Entropy"
)

# ╔═╡ b27a4714-cbda-417e-85e1-26d7d98780ee
dm = Dict([
		"Asymmetric Relative Entropy" => AsymmetricRelativeEntropy(),
		"Lp Entropy" => LpEntropy(),
		"Symmetric Relative Entropy" => SymmetricRelativeEntropy(),
		"Hellinger Distance" => HellingerDistance()
	])

# ╔═╡ 033c9a5d-4a98-48ea-ac01-13f5126bb6f1
md"You can also choose the type of energy map to use."

# ╔═╡ 9eee7238-6e9c-4837-a30b-ebd09abdcca6
@bind e_measure Radio(
	[
		"Time Frequency",
		"Probability Density",
	],
	default = "Time Frequency"
)

# ╔═╡ fd63c142-ae62-40a2-b34f-986c803ddb72
em = Dict(["Time Frequency"=>TimeFrequency(), "Probability Density"=>ProbabilityDensity()])

# ╔═╡ df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
md"You can also choose the number of features you want to extract. You can use the slider below to choose any thing from a single feature to all features (32 in this case)."

# ╔═╡ 9e523918-daaf-4c17-851a-7fac12b04cd3
@bind dim Slider(1:length(X₀[:,1]), default=5, show_value=true)

# ╔═╡ 7a46152b-2df4-41ae-96ff-a4e8a06c1a70
coefs₀, ỹ₀, bt₀, pw₀, ord₀ = ldb(
	X₀, Y₀, 
	wavelet(WT.coif6), 
	dm=dm[d_measure], 
	energy=em[e_measure], 
	m=dim
);

# ╔═╡ 2a9efa07-447e-46d8-b0a6-8560a6765a1f
md"The `ldb` function will return a vector of features sorted by their disciminant power as well as a vector with the discriminant measures as well. We can visually select the number of features to use by creating a scree plot and choosing the *elbow*."

# ╔═╡ d40cf65d-bfac-439f-a83e-5755302f1207
Plots.plot(pw₀, 
		   ylabel = "Discriminant Power", 
	       xlabel = "Number of Features", 
	       label = "")

# ╔═╡ 96a49a0c-4646-43f9-98c2-09ac484f6df8
md"## Signal Classification"

# ╔═╡ 406e7ffe-fa01-4622-ae09-aca5473bfe6c
md"Now we will attempt to classify the 3 classes of signals using LDB. Let us generate 100 training samples and 1000 test samples."

# ╔═╡ 7a9548a4-c528-41af-bba7-a99b0c91247b
begin
	machines = Dict() # Models
	X = (train=Dict(), test=Dict())
	y = (train=Dict(), test=Dict())
	df = DataFrame(Type = String[], trainerr = Float64[], testerr = Float64[])
end;

# ╔═╡ 4774bfcf-9e50-428c-b51f-76a887031862
begin
	X_train, y_train = triangular_test_functions(33, 33, 34, shuffle=true)
	X_test, y_test = triangular_test_functions(333, 333, 334, shuffle=true)
	
	X.train["STD"], X.test["STD"] = X_train', X_test'
	y.train["STD"], y.test["STD"] = coerce(y_train, Multiclass), coerce(y_test, Multiclass)
end;

# ╔═╡ 2d398c73-37bc-44d4-8559-e220de94624d
md"Next, we will load some machine learning models from `MLJ.jl`. We will include two very basic decision tree models(with and without pruning), Linear Discriminant classifier (LDA), Multinomial classifier with L1 regularization (i.e. LASSO), and finally a Random Forest classifier" 

# ╔═╡ 7a7cae84-3272-4303-80fa-d56a8615b9ff
begin
	Tree = @load DecisionTreeClassifier pkg=DecisionTree
	LDA = @load LDA pkg=MultivariateStats
	MCLogit = @load MultinomialClassifier pkg=MLJLinearModels
	RForest = @load RandomForestClassifier pkg=ScikitLearn
end;

# ╔═╡ 54fdabe6-85ff-4928-ac1c-1555d89ce456
md"Intialize the ML models"

# ╔═╡ 0707059e-9f04-42b7-9b6b-a1de72b24a5f
begin
	machines["FCT"] = Tree()
	machines["PCT"] = Tree(post_prune=true, merge_purity_threshold=0.8)
	machines["LDA"] = LDA()
	machines["MCLogit"] = MCLogit(penalty=:l1, lambda=0.01)
	machines["RForest"] = RForest()
end;

# ╔═╡ 51d745c9-8c1a-41ef-8ee6-c5e9c35d39fe
md"### 1. Training models using the original signal"

# ╔═╡ b0e3e833-47d6-493e-bb51-940267e6f85d
md"To evaluate the LDB algorithm's performance we first train the models using the original signals as input (i.e. Euclidean coordinates). To evaluate the training loss, we will perform a 5 fold cross validation."

# ╔═╡ cadd63da-9e77-422b-b568-489ac75a2294
cv = CV(nfolds=5)

# ╔═╡ fded58ea-e7b1-4be1-b952-b7aa1358d5dd
function evaluate_model(model::String, dat::String, X, y)
	name = model * "-" * dat
	
	# Train error
	evalres = evaluate(machines[model], 
					   X.train[dat], y.train["STD"],
					   resampling=CV(nfolds=5, shuffle=true),
					   measure=cross_entropy)
	trainerr = evalres.measurement[1]
	
	mach = machine(machines[model], X.train[dat], y.train["STD"])
	fit!(mach)
	
	# Test error
	ŷ₁ = predict(mach, X.test[dat])
	testerr = LogLoss()(ŷ₁, y.test["STD"]) |> mean
	
	push!(df, Dict(:Type=>name, :trainerr=>trainerr, :testerr=>testerr))
end

# ╔═╡ 19e7d3f3-970d-4d05-9664-8fe23009fb71
for machine in ["FCT", "PCT", "LDA", "MCLogit", "RForest"]
	evaluate_model(machine, "STD", X, y)
end

# ╔═╡ 4ffbeab9-67c5-46a0-8e09-449d91dfa34c
df

# ╔═╡ 97516db4-c019-49a7-b826-64294fd14220
md"### Using LDB-5"

# ╔═╡ c47d64f3-12fc-4628-9162-21980066bd00
md"Next, we significantly reduce the dimensionality of the models by only using the top 5 most discriminant features obtained from LDB."

# ╔═╡ 437d6c74-990b-4147-a0d2-cf4108fd47a4
coefs₁, ỹ₁, bt₁, pw₁, ord₁ = ldb(
	X_train, y_train, 
	wavelet(WT.coif6), 
	dm=dm[d_measure], 
	energy=em[e_measure],
	m=dim
);

# ╔═╡ a828877d-1f49-4b76-b397-869bb11e40c5
begin
	X.train["LDB5"] = coefs₁';
	X.test["LDB5"] = bestbasiscoef(X_test, wavelet(WT.coif6), bt₁)[ord₁,:]';
end;

# ╔═╡ 34ff82ef-e7a7-4df2-ab71-3280a5ef34fe
for machine in ["FCT", "PCT", "LDA", "MCLogit", "RForest"]
	evaluate_model(machine, "LDB5", X, y)
end

# ╔═╡ 407cce96-73cb-4baf-90f9-b46d5d617018
df

# ╔═╡ 7dd079af-0445-436c-9bd3-9550cfaa9b19
md"### 3. Using all LDB features"

# ╔═╡ b31e54a1-f1b7-44c4-b2bc-99123933c289
md"Finally, we use all the LDB features to train our models. Note that we do not include the LDA model because theoretically it is the same with using the euclidean coordinates."

# ╔═╡ 603c25aa-da7c-4d6c-bfae-a64eba519389
coefs₂, ỹ₂, bt₂, pw₂, ord₂ = ldb(X_train, y_train, 
								 wavelet(WT.coif6), 
								 dm=dm[d_measure], 
								 energy=em[e_measure]);

# ╔═╡ d51bddaa-d44c-4b97-acde-483939a6d7f8
begin
	X.train["LDB"] = coefs₂'
	X.test["LDB"] = bestbasiscoef(X_test, wavelet(WT.coif6), bt₂)[ord₂,:]'
end;

# ╔═╡ 3a524156-593f-4a01-91f2-af58e2d75e13
for machine in ["FCT", "PCT", "MCLogit", "RForest"]
	evaluate_model(machine, "LDB", X, y)
end

# ╔═╡ 3f3d4bcf-3f2b-4140-ba52-2246c5140303
df

# ╔═╡ 9b292713-1580-48ae-b9cc-05dca097a673
md"## Results"

# ╔═╡ c1b823e9-4a80-4cca-9527-5b0f2933933d
Gadfly.plot(
	sort(df, :testerr, rev=true),
	layer(x=:testerr, y=:Type, Geom.point, color=[colorant"red"]),
	layer(x=:trainerr, y=:Type, Geom.point, color=[colorant"blue"]),
	Guide.title("Model Performance"),
	Guide.xlabel("Log Loss (Categorical Cross Entropy)"),
	Guide.manual_color_key("",["Train Error","Test Error"], [Gadfly.current_theme().default_color,"red"]))

# ╔═╡ 35c8290b-0c34-49c0-bbef-d1b0e781ee02
md"The plot above shows the performance of each model in terms of Log Loss (a.k.a. Categorical Cross Entropy). The most notable result is that the Multinomial Logit, LDA, and Random Forest models using LDB-5 are performing very well. This demonstrates how LDB can significantly reduce model dimensionality while maitining classification accuracy.\
\
In our experiment, basic decision trees perform terribly. This could be because of overfitting or that they are not suited for modeling this particular data.
"  

# ╔═╡ Cell order:
# ╟─45f88030-a821-11eb-0c6d-f5c7c82b7115
# ╠═45468d3a-3456-4e99-aec8-b3c41b20a426
# ╠═5ad9f0fb-3688-4b15-94c1-a18e5f41eeed
# ╟─f3a892ef-0c5a-4d4a-86ab-036d913d9d97
# ╟─c195f5d9-2538-4278-9d27-c14446e7cb65
# ╟─a751cd87-80c5-48b1-b798-f1aecebc08a1
# ╠═f05ee9bc-3d1f-4ada-a4d7-0f44a5454646
# ╟─b8077eb3-ec64-4a84-9dcc-3aafce015597
# ╠═dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
# ╟─a8f1e20b-e828-4c1b-80d0-999f5c3b6e5e
# ╟─59a3c5b3-d3c6-4b16-ae1b-984b6a77350a
# ╟─39f64d00-350d-43a6-bf57-06600aac2365
# ╟─37a7c05f-eb0b-4509-9945-e482c4b9bc5a
# ╟─28604f68-a957-4a3c-92f5-13a0ff4ba158
# ╟─b27a4714-cbda-417e-85e1-26d7d98780ee
# ╟─033c9a5d-4a98-48ea-ac01-13f5126bb6f1
# ╟─9eee7238-6e9c-4837-a30b-ebd09abdcca6
# ╟─fd63c142-ae62-40a2-b34f-986c803ddb72
# ╟─df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
# ╠═9e523918-daaf-4c17-851a-7fac12b04cd3
# ╠═7a46152b-2df4-41ae-96ff-a4e8a06c1a70
# ╟─2a9efa07-447e-46d8-b0a6-8560a6765a1f
# ╠═d40cf65d-bfac-439f-a83e-5755302f1207
# ╟─96a49a0c-4646-43f9-98c2-09ac484f6df8
# ╟─406e7ffe-fa01-4622-ae09-aca5473bfe6c
# ╠═7a9548a4-c528-41af-bba7-a99b0c91247b
# ╠═4774bfcf-9e50-428c-b51f-76a887031862
# ╟─2d398c73-37bc-44d4-8559-e220de94624d
# ╠═7a7cae84-3272-4303-80fa-d56a8615b9ff
# ╟─54fdabe6-85ff-4928-ac1c-1555d89ce456
# ╠═0707059e-9f04-42b7-9b6b-a1de72b24a5f
# ╟─51d745c9-8c1a-41ef-8ee6-c5e9c35d39fe
# ╟─b0e3e833-47d6-493e-bb51-940267e6f85d
# ╠═cadd63da-9e77-422b-b568-489ac75a2294
# ╠═fded58ea-e7b1-4be1-b952-b7aa1358d5dd
# ╠═19e7d3f3-970d-4d05-9664-8fe23009fb71
# ╟─4ffbeab9-67c5-46a0-8e09-449d91dfa34c
# ╟─97516db4-c019-49a7-b826-64294fd14220
# ╟─c47d64f3-12fc-4628-9162-21980066bd00
# ╠═437d6c74-990b-4147-a0d2-cf4108fd47a4
# ╠═a828877d-1f49-4b76-b397-869bb11e40c5
# ╠═34ff82ef-e7a7-4df2-ab71-3280a5ef34fe
# ╟─407cce96-73cb-4baf-90f9-b46d5d617018
# ╟─7dd079af-0445-436c-9bd3-9550cfaa9b19
# ╟─b31e54a1-f1b7-44c4-b2bc-99123933c289
# ╠═603c25aa-da7c-4d6c-bfae-a64eba519389
# ╠═d51bddaa-d44c-4b97-acde-483939a6d7f8
# ╠═3a524156-593f-4a01-91f2-af58e2d75e13
# ╟─3f3d4bcf-3f2b-4140-ba52-2246c5140303
# ╟─9b292713-1580-48ae-b9cc-05dca097a673
# ╟─c1b823e9-4a80-4cca-9527-5b0f2933933d
# ╟─35c8290b-0c34-49c0-bbef-d1b0e781ee02
