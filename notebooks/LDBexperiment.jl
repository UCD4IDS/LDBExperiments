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
		Wavelets, WaveletsExt, 
		DecisionTree, 
		Plots,
		PlutoUI
	using ScikitLearn.CrossValidation: cross_val_score
end

# ╔═╡ a8a73cee-1113-48ac-83a5-6aa2c46b59f0
include("../src/utils.jl")

# ╔═╡ 45f88030-a821-11eb-0c6d-f5c7c82b7115
md"# Local Discriminant Basis"

# ╔═╡ b8077eb3-ec64-4a84-9dcc-3aafce015597
md"We begin by generate test functions. The `triangluar_test_functions` function will generate a set of test functions consisting of 3 classes."

# ╔═╡ 4e02c2d4-98f5-4c3a-89f5-9c4009a91b5d
X, y = triangular_test_functions(10,10,10);

# ╔═╡ 39f64d00-350d-43a6-bf57-06600aac2365
begin
	p1 = wiggle(X[:,1:5])
	plot!(xlab = "Class 1")
	p2 = wiggle(X[:,11:15])
	plot!(xlab = "Class 2")
	p3 = wiggle(X[:,21:25])
	plot!(xlab = "Class 3")
	plot(p1, p2, p3, layout = (3,1))
end

# ╔═╡ 37a7c05f-eb0b-4509-9945-e482c4b9bc5a
md"Use the `ldb` function from the `WaveletsExt.jl` package. By default, the `ldb` algorithm uses `Assymmetric Relative Entropy` (i.e. Kullback Leiber Divergence) to calculate the distance between the expansion coefficients of different classes of signals. The function will return a set of expansion coefficients with reduce features and dimensions"

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

# ╔═╡ 9e523918-daaf-4c17-851a-7fac12b04cd3
@bind dim Slider(1:length(X[:,1]), default=5, show_value=true)

# ╔═╡ 7a46152b-2df4-41ae-96ff-a4e8a06c1a70
coefs₀, ỹ₀, bt₀, pw₀, ord₀ = ldb(
	X, y, 
	wavelet(WT.coif6), 
	dm=dm[d_measure], 
	energy=em[e_measure], 
	m=dim
);

# ╔═╡ 77c0c0ee-b508-419e-8e72-fafc1085577f
begin
	model = DecisionTreeClassifier(max_depth=5)
	fit!(model, coefs₀', y)
end

# ╔═╡ 18095da6-5a5b-40f2-8421-80b5ec778b67
cross_val_score(model, coefs₀', y, cv=5) |> mean

# ╔═╡ 406e7ffe-fa01-4622-ae09-aca5473bfe6c
md"Lets do it with 100 training samples and 1000 test samples!"

# ╔═╡ 4774bfcf-9e50-428c-b51f-76a887031862
begin
	X_train, y_train = triangular_test_functions(33, 33, 34, shuffle=true)
	X_test, y_test = triangular_test_functions(333, 333, 334, shuffle=true)
end;

# ╔═╡ 437d6c74-990b-4147-a0d2-cf4108fd47a4
coefs₁, ỹ₁, bt₁, pw₁, ord₁ = ldb(
	X_train, 
	y_train, 
	wavelet(WT.coif6), 
	dm=dm[d_measure], 
	energy=em[e_measure],
	m=dim
)

# ╔═╡ 794172e4-0d24-44b0-9c12-f3ddfbb783b4
md"Train the decision tree"

# ╔═╡ 1b4c56d4-8700-4317-abb0-d32d2117931a
begin
	model₁ = DecisionTreeClassifier(max_depth=10)
	fit!(model₁, coefs₁', y_train)
end

# ╔═╡ 4947b60c-e699-421c-b6f8-f243a551cda1
cross_val_score(model₁, coefs₁', y_train, cv=5) |> mean

# ╔═╡ a745518f-0709-40a9-92bb-a6e27755f0f2
begin
	correct = 0
	for i in axes(X_test,2)
		β = bestbasiscoef(X_test[:,i], wavelet(WT.coif6), bt₁)
		pred = predict(model₁, β[ord₁])
		if y_test[i] == pred
			correct += 1
		end
	end
end

# ╔═╡ f99c31c8-d953-47da-9776-ff5cf960db0c
accuracy = correct/length(y_test)

# ╔═╡ Cell order:
# ╟─45f88030-a821-11eb-0c6d-f5c7c82b7115
# ╠═45468d3a-3456-4e99-aec8-b3c41b20a426
# ╠═5ad9f0fb-3688-4b15-94c1-a18e5f41eeed
# ╠═a8a73cee-1113-48ac-83a5-6aa2c46b59f0
# ╟─b8077eb3-ec64-4a84-9dcc-3aafce015597
# ╠═4e02c2d4-98f5-4c3a-89f5-9c4009a91b5d
# ╠═39f64d00-350d-43a6-bf57-06600aac2365
# ╟─37a7c05f-eb0b-4509-9945-e482c4b9bc5a
# ╟─28604f68-a957-4a3c-92f5-13a0ff4ba158
# ╟─b27a4714-cbda-417e-85e1-26d7d98780ee
# ╟─033c9a5d-4a98-48ea-ac01-13f5126bb6f1
# ╟─9eee7238-6e9c-4837-a30b-ebd09abdcca6
# ╟─fd63c142-ae62-40a2-b34f-986c803ddb72
# ╠═9e523918-daaf-4c17-851a-7fac12b04cd3
# ╠═7a46152b-2df4-41ae-96ff-a4e8a06c1a70
# ╠═77c0c0ee-b508-419e-8e72-fafc1085577f
# ╠═18095da6-5a5b-40f2-8421-80b5ec778b67
# ╟─406e7ffe-fa01-4622-ae09-aca5473bfe6c
# ╠═4774bfcf-9e50-428c-b51f-76a887031862
# ╠═437d6c74-990b-4147-a0d2-cf4108fd47a4
# ╟─794172e4-0d24-44b0-9c12-f3ddfbb783b4
# ╠═1b4c56d4-8700-4317-abb0-d32d2117931a
# ╠═4947b60c-e699-421c-b6f8-f243a551cda1
# ╠═a745518f-0709-40a9-92bb-a6e27755f0f2
# ╠═f99c31c8-d953-47da-9776-ff5cf960db0c
