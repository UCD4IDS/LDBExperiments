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

# ╔═╡ 45f88030-a821-11eb-0c6d-f5c7c82b7115
md"# Local Discriminant Basis  
**Author**: Shozen Dan"

# ╔═╡ f3a892ef-0c5a-4d4a-86ab-036d913d9d97
md"## A Brief History"

# ╔═╡ c195f5d9-2538-4278-9d27-c14446e7cb65
md"**Local Discriminant Basis (LDB)** is a wavelet based feature extraction method concieved by Naoki Saito in 1992. Earlier that year, [Victor Wickerhauser](https://www.math.wustl.edu/~victor/) had generalized the best basis algorithm such that it worked not only for a single signal, but for a collection of signals that share the same important features. The so called Joint Best Basis (JBB) can be viewed as a time-frequency localized version of the Principle Component Analysis(PCA) or the Karhunen-Loève Basis (KLB).\
\
While JBB is good for signals belonging to the same class (i.e. share the same features), it does not work for signal classifications in general. LDB sets out to solve this issue by replacing the original minimum entropy cost function used in the JBB with the Kullback-Leibler divergence (a.k.a. relative entropy). More specifically,

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
md"## Short Tutorial"

# ╔═╡ b8077eb3-ec64-4a84-9dcc-3aafce015597
md"We begin by obtaining some signals to classify. Using `ClassData()` alongside `generateclassdata()`, we can generate 2 different sets of signals (Cylinder-Bell-Funnel, Triangular waveform), each consisting of 3 classes of signals."

# ╔═╡ e8182e69-8776-4ab5-a38e-bf2175ceb0ea
md"**Select** the dataset"

# ╔═╡ 910d24a0-8c00-42c5-8e11-13da2557a09d
@bind sigtype Radio(["Cylinder-Bell-Funnel","Triangular"], default = "Cylinder-Bell-Funnel")

# ╔═╡ 705723ac-b0e0-4205-b3aa-8b516f9233d4
st = Dict(["Triangular" => :tri, "Cylinder-Bell-Funnel" => :cbf])

# ╔═╡ dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
begin
	X₀, y₀ = generateclassdata(ClassData(st[sigtype],10,10,10));
	Y₀ = coerce(y₀, Multiclass); # For compatibility with MLJ.jl
end

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

# ╔═╡ 3c077b0c-ad81-43bf-af45-32d7f48185c7
md"**Local Discriminant Basis (LDB)** is a feature extractor meaning that given a set of signals and their labels
$\mathbf{X} = (\mathbf{x_1},\mathbf{x_2},\mathbf{x_3},...), \mathbf{y} = (y_1,y_2,y_3,...)$ where $\mathbf{X} \subset \mathbb{R}^n$ and $y \in \{1,2,...,K\}$, LDB will find a map $f: \mathbf{X} \to \mathbf{Z} \subset \mathbb{R}^m$ such that $m \leq n$.
"

# ╔═╡ 25720fa1-ad95-451d-8143-ba34b6e0551b
md"In the context of LDB, the map $f$ can be expressed as

$f = \Theta_m \circ\Psi^T$

where, $\Psi$ is an $n$-dimensional orthogonal matrix selected from a **library of orthonormal bases**, and $\Theta_m$ is a selection rule that selects the most important $m$ features from $n$ features."

# ╔═╡ e0c92531-1e17-4757-885d-408d62829766
md"After the features have been extracted, we can use a conventional classifier such as Linear Discriminant Anlaysis (LDA), Classfication Trees (CART), or Multiclass Logistic Regression to map $\mathbf{Z} \to \mathbf{y}$."

# ╔═╡ dc5ba00d-1a5b-4233-96a6-73981882345a
md"### Library of Orthonormal Bases"

# ╔═╡ 01a254f4-9575-4ab2-af6a-27ad5ef8efde
md"The LDB algorithm begins by decomposing each signal into a dictionary of orthonormal bases. Each dictionary is a binary tree where each node is a subspace of $\mathbf{w}_{0,0} = \mathbb{R}^n$. Examples of dictionaries include wavelet packet bases, local trigonometric bases, and local Fourier bases. In the figure below, we illustrate the wavelet packet bases where $G$, $H$, and $f$ represent the scaling (lowpass) analysis filters, wavelet (highpass) analysis filters, and the frequency axis respectively. By decomposing a signal as such, information about the signal at a specific time and frequency can be encoded within each node, which can then be used to distinguish different classes of signals."

# ╔═╡ 9f7c2639-c455-425a-a2ab-0deac638b47f
img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/wpt-figure.png";

# ╔═╡ b2db449c-0fe5-482a-9e85-9062a218df03
md"""$(Resource(img_url, :width => 500))"""

# ╔═╡ 964f8fcd-0516-4c6f-a02a-6db5dd497520
em_img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/normalized-energy-map.png";

# ╔═╡ 1c4b794f-2b17-429d-809a-2f69f0a82e41
md"### Energy Map"

# ╔═╡ a3b05137-0499-45c1-bbce-79784dbf59dc
md"**Normalized energies**"

# ╔═╡ 55528ce0-3374-4d13-bb6f-61df9f854a39
md"
$V_i^{(y)} \triangleq 
\frac{
	E\left[Z_i^2|Y=y\right]
}{
	\sum_{i=1}^n E\left[Z_i^2|Y=y\right]
}
\to 
\hat{V}_i^{(y)} = 

\frac{
	\sum_{k=1}^{N_y} \left\| \boldsymbol{w}_i \cdot \boldsymbol{x}_k^{(y)}\right\|^2
}{
	\sum_k^{N_y} \left\| \boldsymbol{x}_k^{(y)} \right\|^2
}$"

# ╔═╡ 8d2707f8-a982-4c83-b14a-1a2deb6743b4
md"""$(Resource(em_img_url))"""

# ╔═╡ d9973444-b859-4377-bcf0-2c6885933380
pem_img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/probability-energy-map.png";

# ╔═╡ fb91da71-303f-4c43-be7b-e39df1429355
md"**Probability density**

Another way to estimate $E\left[Z_i^2|Y=y\right]$ is to use kernel density estimators. The LDB algorithm in `WaveletsExt.jl` uses a method called Average Shifted Histograms(ASH)."

# ╔═╡ af1b235e-6fff-478f-a5c1-38fbc6c39b8f
md"
$q_i^{(y)}(z) \triangleq 

\int_{\boldsymbol{w}_i \cdot \boldsymbol{x}=\boldsymbol{z}}
	p(\boldsymbol{x}|y)d\boldsymbol{x} \to \hat{q}_i^{(y)}(z)$"

# ╔═╡ d27634a5-e703-4fa2-bc1a-d7297b2388a3
md"""$(Resource(pem_img_url))"""

# ╔═╡ e0a525ed-35f0-48cc-8403-ddfe03871074
md"**Select** the type of energy map to use"

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

# ╔═╡ 4f8a7bb5-db64-4f82-8544-c961cca068db
md"### Discriminant Measure"

# ╔═╡ e77667ca-9bb8-4f30-b5ba-ff107eb9a568
md"Asymmetric Relative Entropy (Kullback-Leibler divergence):

$D_{KL}(p,q) = \int_{-\infty}^{\infty}p(x)\log_2\frac{p(x)}{q(x)}dx$"

# ╔═╡ ed92e98f-e823-45a6-a401-342f584c333e
md" $L^P$ entropy

$D_P(p,q) = \left(\int_{-\infty}^{\infty}(p(x)^P - q(x)^P)dx\right)^{1/P}$"

# ╔═╡ 0b12ee12-9229-486f-aa65-1da5c53955cc
md"Hellinger distance

$D_H(p,q)=\int_{-\infty}^{\infty}\left(\sqrt{p(x)} - \sqrt{q(x)}\right)^2dx$" 

# ╔═╡ 885cc8dd-dc5a-4d28-be72-2e26613ec252
md"Symmetric Relative Entropy

$$D_S(p,q) = D_{KL}(p,q) + D_{KL}(q,p)$$
"

# ╔═╡ 05a9e8db-fce0-4d12-b67b-c0089621ae07
md"**Select** the type of discriminant measure to use"

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

# ╔═╡ df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
md"You can also choose the number of features you want to extract. You can use the slider below to choose any thing from a single feature to all features (32 in this case)."

# ╔═╡ f1c6268b-a6d5-445a-9e52-748898ec08da
md"**Select** the number of features to extract"

# ╔═╡ 9e523918-daaf-4c17-851a-7fac12b04cd3
@bind dim Slider(1:length(X₀[:,1]), default=10, show_value=true)

# ╔═╡ 2c9b5aef-ba63-44b6-8ef9-ca31cc682fad
ldb₀ = LocalDiscriminantBasis(wt = wavelet(WT.coif6),
					   		  dm = dm[d_measure],
					   		  en = em[e_measure],
							  n_features = dim)

# ╔═╡ 2a9efa07-447e-46d8-b0a6-8560a6765a1f
md"The `ldb` function will return a vector of features sorted by their disciminant power as well as a vector with the discriminant measures as well. We can visually select the number of features to use by creating a scree plot and choosing the *elbow*."

# ╔═╡ d55eb3ed-cf38-4f51-8245-fdb427312eb8
WaveletsExt.fit!(ldb₀, X₀, y₀)

# ╔═╡ 0e4f9614-6972-4993-9d65-f4cf515553bd
md"The plot below shows represents the best basis tree (dictionary of orthogonal bases) that maximizes the discriminant measure between the 3 classes of signals."

# ╔═╡ 6d968714-1058-4771-8964-20621ca9ffc6
plot_tfbdry(ldb₀.tree)

# ╔═╡ 3ece6ff2-adaf-476b-bd01-dbd48dc00f15
md"Normalized energy map/probability density of each class."

# ╔═╡ f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
begin
	hmap1 = Plots.heatmap(ldb₀.Γ[:,:,1]',yflip=true,xlabel="Class 1",legend=false);
	hmap2 = Plots.heatmap(ldb₀.Γ[:,:,2]',yflip=true,xlabel="Class 2",legend=false);
	hmap3 = Plots.heatmap(ldb₀.Γ[:,:,3]',yflip=true, xlabel="Class 3",legend=false);
	Plots.plot(hmap1, hmap2, hmap3, layout=(3,1))
end

# ╔═╡ 121fd299-5e82-4159-8472-5d385c736c18
Plots.heatmap(discriminant_measure(ldb₀.Γ, dm[d_measure])',yflip=true)

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
	X_train, y_train = generateclassdata(ClassData(st[sigtype],33,33,33), true)
	X_test, y_test = generateclassdata(ClassData(st[sigtype],333,333,333), true)
	
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
	MLJ.fit!(mach)
	
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

# ╔═╡ a828877d-1f49-4b76-b397-869bb11e40c5
begin
	ldb = LocalDiscriminantBasis(wt = wavelet(WT.coif6),
								 dm = dm[d_measure],
								 en = em[e_measure],
	                             n_features = 10)
	WaveletsExt.fit!(ldb, X_train, y_train)
	X.train["LDB5"] = WaveletsExt.transform(ldb, X_train)';
	X.test["LDB5"] = WaveletsExt.transform(ldb, X_test)';
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

# ╔═╡ a08dd1ea-a403-41b4-915c-56fde82222e7
ldb₁ = LocalDiscriminantBasis(
	wt = wavelet(WT.coif6),
	dm = dm[d_measure],
	en = em[e_measure]
)

# ╔═╡ 3a374ac2-e225-4637-9dbd-6644cb80b629
WaveletsExt.fit!(ldb₁, X_train, y_train)

# ╔═╡ d51bddaa-d44c-4b97-acde-483939a6d7f8
begin
	X.train["LDB"] = WaveletsExt.transform(ldb₁, X_train)'
	X.test["LDB"] = WaveletsExt.transform(ldb₁, X_test)'
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
# ╟─b8077eb3-ec64-4a84-9dcc-3aafce015597
# ╟─e8182e69-8776-4ab5-a38e-bf2175ceb0ea
# ╟─910d24a0-8c00-42c5-8e11-13da2557a09d
# ╟─705723ac-b0e0-4205-b3aa-8b516f9233d4
# ╟─dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
# ╟─59a3c5b3-d3c6-4b16-ae1b-984b6a77350a
# ╟─39f64d00-350d-43a6-bf57-06600aac2365
# ╟─3c077b0c-ad81-43bf-af45-32d7f48185c7
# ╟─25720fa1-ad95-451d-8143-ba34b6e0551b
# ╟─e0c92531-1e17-4757-885d-408d62829766
# ╟─dc5ba00d-1a5b-4233-96a6-73981882345a
# ╟─01a254f4-9575-4ab2-af6a-27ad5ef8efde
# ╟─9f7c2639-c455-425a-a2ab-0deac638b47f
# ╟─b2db449c-0fe5-482a-9e85-9062a218df03
# ╟─964f8fcd-0516-4c6f-a02a-6db5dd497520
# ╟─1c4b794f-2b17-429d-809a-2f69f0a82e41
# ╟─a3b05137-0499-45c1-bbce-79784dbf59dc
# ╟─55528ce0-3374-4d13-bb6f-61df9f854a39
# ╟─8d2707f8-a982-4c83-b14a-1a2deb6743b4
# ╟─d9973444-b859-4377-bcf0-2c6885933380
# ╟─fb91da71-303f-4c43-be7b-e39df1429355
# ╟─af1b235e-6fff-478f-a5c1-38fbc6c39b8f
# ╟─d27634a5-e703-4fa2-bc1a-d7297b2388a3
# ╟─e0a525ed-35f0-48cc-8403-ddfe03871074
# ╟─9eee7238-6e9c-4837-a30b-ebd09abdcca6
# ╟─fd63c142-ae62-40a2-b34f-986c803ddb72
# ╟─4f8a7bb5-db64-4f82-8544-c961cca068db
# ╟─e77667ca-9bb8-4f30-b5ba-ff107eb9a568
# ╟─ed92e98f-e823-45a6-a401-342f584c333e
# ╟─0b12ee12-9229-486f-aa65-1da5c53955cc
# ╟─885cc8dd-dc5a-4d28-be72-2e26613ec252
# ╟─05a9e8db-fce0-4d12-b67b-c0089621ae07
# ╟─28604f68-a957-4a3c-92f5-13a0ff4ba158
# ╟─b27a4714-cbda-417e-85e1-26d7d98780ee
# ╟─df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
# ╟─f1c6268b-a6d5-445a-9e52-748898ec08da
# ╟─9e523918-daaf-4c17-851a-7fac12b04cd3
# ╠═2c9b5aef-ba63-44b6-8ef9-ca31cc682fad
# ╟─2a9efa07-447e-46d8-b0a6-8560a6765a1f
# ╠═d55eb3ed-cf38-4f51-8245-fdb427312eb8
# ╟─0e4f9614-6972-4993-9d65-f4cf515553bd
# ╠═6d968714-1058-4771-8964-20621ca9ffc6
# ╟─3ece6ff2-adaf-476b-bd01-dbd48dc00f15
# ╟─f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
# ╠═121fd299-5e82-4159-8472-5d385c736c18
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
# ╠═fded58ea-e7b1-4be1-b952-b7aa1358d5dd
# ╠═19e7d3f3-970d-4d05-9664-8fe23009fb71
# ╟─4ffbeab9-67c5-46a0-8e09-449d91dfa34c
# ╟─97516db4-c019-49a7-b826-64294fd14220
# ╟─c47d64f3-12fc-4628-9162-21980066bd00
# ╠═a828877d-1f49-4b76-b397-869bb11e40c5
# ╠═34ff82ef-e7a7-4df2-ab71-3280a5ef34fe
# ╟─407cce96-73cb-4baf-90f9-b46d5d617018
# ╟─7dd079af-0445-436c-9bd3-9550cfaa9b19
# ╟─b31e54a1-f1b7-44c4-b2bc-99123933c289
# ╠═a08dd1ea-a403-41b4-915c-56fde82222e7
# ╠═3a374ac2-e225-4637-9dbd-6644cb80b629
# ╠═d51bddaa-d44c-4b97-acde-483939a6d7f8
# ╠═3a524156-593f-4a01-91f2-af58e2d75e13
# ╟─3f3d4bcf-3f2b-4140-ba52-2246c5140303
# ╟─9b292713-1580-48ae-b9cc-05dca097a673
# ╟─c1b823e9-4a80-4cca-9527-5b0f2933933d
# ╟─35c8290b-0c34-49c0-bbef-d1b0e781ee02
