### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
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

# ╔═╡ aa7ba1e0-ff23-4406-96a1-4406b4e94399
begin
	include("./src/mod/data.jl")
	include("./src/mod/plots.jl")
	import .LDBDatasets: get_dataset, textures_label
	import .LDBPlots: plot_emap, plot_emaps
end

# ╔═╡ 45f88030-a821-11eb-0c6d-f5c7c82b7115
md"# Local Discriminant Basis  
**Authors**: Shozen Dan, Zeng Fung Liew, Naoki Saito"

# ╔═╡ f3a892ef-0c5a-4d4a-86ab-036d913d9d97
md"## I. Introduction"

# ╔═╡ c195f5d9-2538-4278-9d27-c14446e7cb65
md"**Local Discriminant Basis (LDB)** is a wavelet based feature extraction method concieved by [Naoki Saito](https://www.math.ucdavis.edu/~saito). In 1992, [Victor Wickerhauser](https://www.math.wustl.edu/~victor/) had generalized the wavelet best basis algorithm such that it worked not only for a single signal, but for a collection of signals that share the same features. This so called Joint Best Basis (JBB) can be viewed as a time-frequency localized version of the Principle Component Analysis (PCA) or the Karhunen-Loève Basis (KLB).\
\
While JBB is good for signals belonging to the same class, it does not work for signal classifications in general. LDB sets out to solve this issue by replacing the original minimum entropy cost function used in the JBB with discriminant measures such as Kullback-Leibler divergence. In a nutshell, the LDB takes to following steps to extract discriminant features from a set of signals.

1. Decomposes individual signals into time-frequency dictionaries (wavelet packet decomposition). 
2. Creates a time-frequency energy distribution for each class by accumulating these dictionaries.
3. A complete orthonormal basis called LDB, which encodes the time-frequency localized similarity and differences between signal classes, is computed using discriminant measures such as KL-divergence.
4. The coordinates that house the most distinguishing (the most divergent) feature of each signal class is identified.
5. The features at these coordinates are extracted from the time-frequency dictionaries of each individual signal and fed to classification algorithms such as Decision Trees or Linear Discriminant Analysis.\

Some resourses on LDB including the original paper can be found in the following links:
* [Local Discriminant Basis[1]](https://www.math.ucdavis.edu/~saito/contrib/#ldb)
* [On Local Orthonormal Bases for Classification and Regression[2]](http://math.ucdavis.edu/~saito/publications/saito_icassp95.pdf)
* [Local discriminant bases and their applications[3]](http://math.ucdavis.edu/~saito/publications/saito_ldb_jmiv.pdf)

The application capabilities of LDB has been studied in multiple studies, with a large number belonging to the bio-medical and neuroscience fields. Results have shown that LDB can successfully classify tumors (e.g. breast and brain), fungal infections, and brain waves using signals obtained from medical sensing techniques such as MRI and EEG[4][5][6].

**References**

[1]N. Saito and R. R. Coifman, “On local orthonormal bases for classification and regression,” in 1995 International Conference on Acoustics, Speech, and Signal Processing, May 1995, vol. 3, pp. 1529–1532 vol.3. doi: 10.1109/ICASSP.1995.479852.

[2]N. Saito, “Naoki’s Selected Scientific Contributions,” Naoki’s Selected Scientific Contributions. https://www.math.ucdavis.edu/~saito/contrib/#ldb

[3]N. Saito and R. R. Coifman, “Local discriminant bases and their applications,” J Math Imaging Vis, vol. 5, no. 4, pp. 337–358, Dec. 1995, doi: 10.1007/BF01250288.

[4]D. Li, W. Pedrycz, and N. J. Pizzi, “Fuzzy wavelet packet based feature extraction method and its application to biomedical signal classification,” IEEE Transactions on Biomedical Engineering, vol. 52, no. 6, pp. 1132–1139, Jun. 2005, doi: 10.1109/TBME.2005.848377.

[5]N. F. Ince, S. Arica, and A. Tewfik, “Classification of single trial motor imagery EEG recordings with subject adapted non-dyadic arbitrary time–frequency tilings,” J. Neural Eng., vol. 3, no. 3, pp. 235–244, Sep. 2006, doi: 10.1088/1741-2560/3/3/006.

[6]S. K. Davis, B. D. Van Veen, S. C. Hagness, and F. Kelcz, “Breast Tumor Characterization Based on Ultrawideband Microwave Backscatter,” IEEE Transactions on Biomedical Engineering, vol. 55, no. 1, pp. 237–246, Jan. 2008, doi: 10.1109/TBME.2007.900564.
"

# ╔═╡ a751cd87-80c5-48b1-b798-f1aecebc08a1
md"## II. Tutorial"

# ╔═╡ 4e5fe030-8a87-4f4a-88a4-b7b824157880
md"**Auto run:** Check the box to start the tutorial

$(@bind autorun CheckBox(default=true))"

# ╔═╡ b8077eb3-ec64-4a84-9dcc-3aafce015597
md"We begin by obtaining some signals to classify. 

For 1D signals, a wrapper function for `WaveletsExt.generateclassdata()` is written for easier signal generations. Here, we can generate 2 different sets of signals (Cylinder-Bell-Funnel, Triangular waveform), each consisting of 3 classes of signals.

For 2D signals, one set of data is obtained from the [SIPI Image Database - Textures](https://sipi.usc.edu/database/database.php?volume=textures). Another set of data used in this experiment is the MNIST dataset, obtained from the MLDatasets.jl package."

# ╔═╡ e8182e69-8776-4ab5-a38e-bf2175ceb0ea
md"**Select** the dataset"

# ╔═╡ 910d24a0-8c00-42c5-8e11-13da2557a09d
@bind sigtype Radio(["Cylinder-Bell-Funnel","Triangular", "Textures", "MNIST"], default = "Cylinder-Bell-Funnel")

# ╔═╡ a0ae476e-7939-4bfe-83e4-9666d0ed366e
md"""
**Classes of textures for texture dataset**

*Ignore the message below if the selected dataset is not Textures. Otherwise, hold `Ctrl` + `Click` on the texture files to generate samples from. Note that each file selected corresponds to a unique class.*

$(sigtype != "Textures" || @bind textclass MultiSelect((sort ∘ collect ∘ keys)(LDBDatasets.textures_label), 
	default = ["Brodatz - Grass (D9) - 512", 
			   "Brodatz - Beach sand (D29) - 512",
			   "Brodatz - Plastic bubbles (D112) - 512"]))
"""

# ╔═╡ 705723ac-b0e0-4205-b3aa-8b516f9233d4
st = Dict(["Triangular" => :tri, "Cylinder-Bell-Funnel" => :cbf,
		   "Textures" => :textures, "MNIST" => :mnist]);

# ╔═╡ dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
begin
	if sigtype ∈ ["Triangular", "Cylinder-Bell-Funnel"]
		args = [5, 0]
	elseif sigtype == "Textures"
		args = [textclass, 1, 0]
	else
		args = [1, 0]
	end
	(X₀, y₀), _ = get_dataset(st[sigtype], args...)
	Y₀ = coerce(y₀, Multiclass); # For compatibility with MLJ.jl
end;

# ╔═╡ 39f64d00-350d-43a6-bf57-06600aac2365
begin
	gr(size=(700,400))
	if sigtype ∈ ["Triangular", "Cylinder-Bell-Funnel"]
		p1 = wiggle(X₀[:,1:5], sc=0.5)
		Plots.plot!(xlab = "Class 1")
		p2 = wiggle(X₀[:,6:10], sc=0.5)
		Plots.plot!(xlab = "Class 2")
		p3 = wiggle(X₀[:,11:15], sc=0.5)
		Plots.plot!(xlab = "Class 3")
		Plots.plot(p1, p2, p3, layout = (3,1))
	else
		plts = []
		for (x, y) in zip(eachslice(X₀, dims=3), y₀)
			plt = heatmap(x', color = :greys, title = y, legend = false, yflip = true,
						  titlefontsize = 7, tickfontsize=5)
			push!(plts, plt)
		end
		Plots.plot(plts...)
	end
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

# ╔═╡ fb91da71-303f-4c43-be7b-e39df1429355
md"**Probability density**

Another way to estimate $E\left[Z_i^2|Y=y\right]$ is to use kernel density estimators. The LDB algorithm in `WaveletsExt.jl` uses a method called Average Shifted Histograms(ASH)."

# ╔═╡ af1b235e-6fff-478f-a5c1-38fbc6c39b8f
md"
$q_i^{(y)}(z) \triangleq 

\int_{\boldsymbol{w}_i \cdot \boldsymbol{x}=\boldsymbol{z}}
	p(\boldsymbol{x}|y)d\boldsymbol{x} \to \hat{q}_i^{(y)}(z)$"

# ╔═╡ d27634a5-e703-4fa2-bc1a-d7297b2388a3
begin
	pem_img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/probability-density-tensor.png";
	
	md"""$(Resource(pem_img_url))"""
end

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
em = Dict(["Time Frequency"=>TimeFrequency(), "Probability Density"=>ProbabilityDensity()]);

# ╔═╡ 4f8a7bb5-db64-4f82-8544-c961cca068db
md"### Discriminant Measure"

# ╔═╡ a0525192-f1d9-4173-960e-ea3c009e067b
md"For each cell within the energy map, we can quatify the difference between each pair of classes by choosing an appropriate chosen discriminant measure. `WaveletsExt.jl` implements four different types of discriminant measures: asymmetic relative entropy, symmetric relative entropy, squared euclidean distance, and Hellinger distance. The definition for each type of discriminant measure is shown below."

# ╔═╡ e77667ca-9bb8-4f30-b5ba-ff107eb9a568
md"Asymmetric Relative Entropy (Kullback-Leibler divergence):

$D_{KL}(p,q) = \int_{-\infty}^{\infty}p(x)\log_2\frac{p(x)}{q(x)}dx$"

# ╔═╡ 885cc8dd-dc5a-4d28-be72-2e26613ec252
md"Symmetric Relative Entropy

$$D_S(p,q) = D_{KL}(p,q) + D_{KL}(q,p)$$
"

# ╔═╡ ed92e98f-e823-45a6-a401-342f584c333e
md"Squared Euclidean Distance

$D_P(p,q) = \int_{-\infty}^{\infty}\left(p(x) - q(x)\right)^2dx$"

# ╔═╡ 0b12ee12-9229-486f-aa65-1da5c53955cc
md"Hellinger distance

$D_H(p,q)=\int_{-\infty}^{\infty}\left(\sqrt{p(x)} - \sqrt{q(x)}\right)^2dx$" 

# ╔═╡ 05a9e8db-fce0-4d12-b67b-c0089621ae07
md"**Select** the type of discriminant measure to use"

# ╔═╡ 28604f68-a957-4a3c-92f5-13a0ff4ba158
@bind d_measure Radio(
	[
		"Asymmetric Relative Entropy",
		"Symmetric Relative Entropy",
		"Squared Euclidean Distance",
		"Hellinger Distance"
	],
	default = "Asymmetric Relative Entropy"
)

# ╔═╡ b27a4714-cbda-417e-85e1-26d7d98780ee
dm = Dict([
		"Asymmetric Relative Entropy" => AsymmetricRelativeEntropy(),
		"Squared Euclidean Distance" => LpDistance(),
		"Symmetric Relative Entropy" => SymmetricRelativeEntropy(),
		"Hellinger Distance" => HellingerDistance()
	]);

# ╔═╡ 7dc1ba75-fa54-4d59-9b54-93e01da7211e
dmm_img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/discriminant-measure-map.png";

# ╔═╡ 86df0e34-8f77-4d6e-8f40-6f4d8e706c15
md"By adding the discriminant measures between each pair of classes together, we can obtain a matrix where the most discriminant cells (i.e. features) have the largest values and the least discriminant cells have the lowest values."

# ╔═╡ ad3fe2dc-8003-451c-bf83-c3c7f24e7f0b
md"""$(Resource(dmm_img_url))"""

# ╔═╡ f0c4b67e-208e-41a4-9510-c47e04a65e20
prn_img_url = "https://raw.githubusercontent.com/ShozenD/LDBExperiments/main/images/pruning.png";

# ╔═╡ 9c4cf3a1-cd6a-4a42-acce-ceaca6c66df2
md"We can use this discriminant measure matrix to prune our binary tree. The pruning algorithm will start from the bottom of the tree and will eliminate the children nodes if the sum of their discriminant measure is smaller or equal to the discriminant measure of their parent node. This pruning process can not only identify features that are most disciminant, but also reduce the computational resources required for to perform LDB."

# ╔═╡ 299c91a6-c315-4299-97fc-57a6b6f419b5
md"""$(Resource(prn_img_url))"""

# ╔═╡ df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
md"After the pruning, we can sort the coefficients by their discriminant power and extract the top n features (wavelet coefficient) to perform dimensionality reduction. We can pass the selected features to ML classification algorithms such as logistic regression, support vector machine, random forests, etc."

# ╔═╡ f1c6268b-a6d5-445a-9e52-748898ec08da
md"**Select** the number of features to extract"

# ╔═╡ 9e523918-daaf-4c17-851a-7fac12b04cd3
begin
	sz = ndims(X₀) == 2 ? size(X₀,1) : size(X₀,1) * size(X₀,2)
	@bind dim Slider(1:sz, default=10, show_value=true)
end

# ╔═╡ 22a8dfdb-c046-4614-8e2b-aab22d12b205
md"To run the LDB algorithm using `WaveletsExt.jl`, we begin by calling `LocalDiscriminantBasis` constructor to specify the parameters. We need to pass 4 arguments to the constructor:

1. The type of wavelet filter to use
2. The type of discriminant measure to use
3. The type of energy map to use
4. The number of features to extract (can be changed at a later stage)"

# ╔═╡ 09bc8c83-2f25-44a0-81ab-9f0a5724673f
md"After the model parameters are specified, the LDB model is trained using the `fit!` method. This follows the syntax in `MLJ.jl`."

# ╔═╡ 2c9b5aef-ba63-44b6-8ef9-ca31cc682fad
if autorun
	ldb₀ = LocalDiscriminantBasis(wt = wavelet(WT.coif6),
								  dm = dm[d_measure],
								  en = em[e_measure],
								  n_features = dim)
	WaveletsExt.fit!(ldb₀, X₀, y₀)
end

# ╔═╡ ae624404-0770-41e9-962b-139006356ea6
md"`WaveletsExt.jl` provides interfaces to understand the results of LDB. For example we can visualize the energy/probability density maps for each class by accessing the `Γ` attribute in the fitted LDB object. The plots below shows the energy/probability density maps for each class, where darker colors indicate that the sub-band has a higher discriminant power.

*Note: When probability density maps are selected, a slider will appear below the energy maps below. Use the slider to select the position of the probability density function and compare the energy maps between classes.*"

# ╔═╡ f1727ea3-7353-4c6a-bc13-51622c615780
if autorun && ldb₀.en == ProbabilityDensity()
	pdf_len = st[sigtype] ∈ [:tri, :cbf] ? size(ldb₀.Γ,3) : size(ldb₀.Γ,4)
	@bind i Slider(1:pdf_len)
end

# ╔═╡ f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
begin
	if autorun
		if ldb₀.en == TimeFrequency()
			st[sigtype]∈[:tri,:cbf] ? gr(size=(700,600)) : gr(size=(600,1800))
			plot_emaps(ldb₀.Γ, ldb₀.tree)
		elseif ldb₀.en == ProbabilityDensity() && st[sigtype] ∈ [:tri, :cbf]
			gr(size=(700,600))
			plot_emaps(ldb₀.Γ[:,:,i,:], ldb₀.tree)
		elseif ldb₀.en == ProbabilityDensity() && st[sigtype] ∈ [:textures, :mnist]
			gr(size=(600,1800))
			plot_emaps(ldb₀.Γ[:,:,:,i,:], ldb₀.tree)
		end
	end
end

# ╔═╡ 82ffc65d-54ea-42ae-a7ef-dbe6216b0d1e
md"We can also visualize the discriminant measure map and the selected nodes/sub-bands using the `discriminant_measure` function and the `tree` attributes respectively. The first plot in the figure below displays the heatmap for the discriminant measure map and the second plot displays the selected nodes from the binary tree."

# ╔═╡ 121fd299-5e82-4159-8472-5d385c736c18
begin
	if autorun
		st[sigtype]≠:textures ? gr(size=(700,400)) : gr(size=(700,1800))
		dmap = plot_emap(discriminant_measure(ldb₀.Γ, dm[d_measure]), ldb₀.tree)
		tmap = st[sigtype]∈[:tri,:cbf] ? plot_tfbdry(ldb₀.tree) : plot_tfbdry2(ldb₀.tree)
		Plots.plot(dmap, tmap, layout=(2,1))
	end
end

# ╔═╡ 96a49a0c-4646-43f9-98c2-09ac484f6df8
md"## III. Signal Classification Experiment"

# ╔═╡ 406e7ffe-fa01-4622-ae09-aca5473bfe6c
md"In this section, we will evaluate the classification capabilities of LDB via a simulation experiment. We will generate 3 classes of signals, use LDB to extract the most discriminant features, and classify the signals using various ML methods."

# ╔═╡ a5126ad3-19b1-4b4e-b96f-ef6b5220854b
md"**Select** dataset"

# ╔═╡ f9a60263-7ebd-4df8-b33f-5f0e85719186
@bind sigtype2 Radio(["Cylinder-Bell-Funnel","Triangular", "Textures", "MNIST"], default = "Cylinder-Bell-Funnel")

# ╔═╡ 6fef2648-058a-4136-8108-38c1624a19ca
begin
	if st[sigtype2] == :tri
		@bind dim2 Slider(1:32, default=10, show_value=true)
	elseif st[sigtype2] == :cbf
		@bind dim2 Slider(1:128, default=10, show_value=true)
	elseif st[sigtype2] == :textures
		@bind dim2 Slider(1:128^2, default=10, show_value=true)
	elseif st[sigtype2] == :mnist
		@bind dim2 Slider(1:32^2, default=10, show_value=true)
	end
end

# ╔═╡ 8f8bb837-ed86-4a75-8254-913530ed8bc5
md"""
**Classes of textures for texture dataset**

*Ignore the message below if the selected dataset is not Textures. Otherwise, hold `Ctrl` + `Click` on the texture files to generate samples from. Note that each file selected corresponds to a unique class.*

$(sigtype2 != "Textures" || @bind textclass2 MultiSelect((sort ∘ collect ∘ keys)(LDBDatasets.textures_label), 
	default = ["Brodatz - Grass (D9) - 512", 
			   "Brodatz - Beach sand (D29) - 512",
			   "Brodatz - Plastic bubbles (D112) - 512"]))
"""

# ╔═╡ 2999257a-03bf-4313-8dd6-d2da0ed2ef9c
md"""**Select** the type of wavelet to use

$(@bind wavelet_type Select(
	["WT.haar", 
	"WT.db1", "WT.db2", "WT.db3", "WT.db4", "WT.db5", 
	"WT.db6", "WT.db7", "WT.db8", "WT.db9", "WT.db10",
	"WT.coif2", "WT.coif4", "WT.coif6", "WT.coif8",
	"WT.sym4", "WT.sym5", "WT.sym6", "WT.sym7", "WT.sym8", "WT.sym9", "WT.sym10",
	"WT.batt2", "WT.batt4", "WT.batt6"],
	default = "WT.coif6"
))"""

# ╔═╡ 65d45fbd-09bf-49e9-b027-e43751ce070f
md"**Select** the type of energy map"

# ╔═╡ bf8314d6-eb38-4594-afb0-eed5f3151389
@bind e_measure2 Radio(
	[
		"Time Frequency",
		"Probability Density",
	],
	default = "Time Frequency"
)

# ╔═╡ dde5cc7d-1840-49a9-bcd0-bf3ed6e66007
md"**Select** the type of discriminant measure"

# ╔═╡ 1b71478b-a386-416b-97dc-a2e5da1ce071
@bind d_measure2 Radio(
	[
		"Asymmetric Relative Entropy",
		"Symmetric Relative Entropy",
		"Lp Entropy",
		"Hellinger Distance"
	],
	default = "Asymmetric Relative Entropy"
)

# ╔═╡ 2bf768ee-7bc7-4039-85d2-dbdf9ed1f75a
md"**Select** the numbers of features to extract"

# ╔═╡ 60cffbcf-d539-4038-9a12-c40fa41d6880
md"**Auto Run**: Check the box after you are satisfied with the experiment parameters or when you want to re-run the experiment (uncheck and check again)

$(@bind autorun2 CheckBox(default=true))"

# ╔═╡ 7a9548a4-c528-41af-bba7-a99b0c91247b
begin
	if autorun2
		dfm = DataFrame(name=String[], model=[], predtype=Symbol[])
		machines = Dict() # Models
		X = (train=Dict(), test=Dict())
		y = (train=Dict(), test=Dict())
		df = DataFrame(Type = String[], TrainAcc = Float64[], TestAcc = Float64[])
	end
end;

# ╔═╡ 4774bfcf-9e50-428c-b51f-76a887031862
begin
	if autorun2
		if sigtype2 ∈ ["Triangular", "Cylinder-Bell-Funnel", "MNIST"]
			args2 = [33, 333]
		else
			args2 = [textclass2, 33, 333]
		end
		(X_train, y_train), (X_test, y_test) = get_dataset(st[sigtype2], args2...)

		nclass_train, nclass_test = (length∘unique)(y_train), (length∘unique)(y_test)
		X.train["STD"]= DataFrame(reshape(X_train,:,33*nclass_train)', :auto)
		X.test["STD"] = DataFrame(reshape(X_test,:,333*nclass_test)', :auto)
		y.train["STD"], y.test["STD"] = coerce(y_train, Multiclass), coerce(y_test, Multiclass)
	end
end;

# ╔═╡ 2d398c73-37bc-44d4-8559-e220de94624d
md"Next, we will load some machine learning models from `MLJ.jl`. We will include two very basic decision tree models(with and without pruning), Linear Discriminant classifier (LDA), Multinomial classifier with L1 regularization (i.e., LASSO), and finally a Random Forest classifier." 

# ╔═╡ 7a7cae84-3272-4303-80fa-d56a8615b9ff
begin
	Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
	LDA = @load LDA pkg=MultivariateStats verbosity=0
	MCLogit = @load MultinomialClassifier pkg=MLJLinearModels verbosity=0
	RForest = @load RandomForestClassifier pkg=DecisionTree verbosity=0
	SVC = @load SVC pkg=LIBSVM verbosity=0
end;

# ╔═╡ 54fdabe6-85ff-4928-ac1c-1555d89ce456
md"Intialize the ML models"

# ╔═╡ 9ddb4726-5a78-4adf-a3eb-0796636467c1
begin
	function add_model!(df::DataFrame, 
						name::String, 
						model, 
						predtype=info(model).prediction_type)

		M = Dict(:name=>name, :model=>model, :predtype=>predtype)
		push!(df, M)
	end

	if autorun2
		add_model!(dfm, "FCT", Tree())
		add_model!(dfm, "PCT", Tree(post_prune=true, merge_purity_threshold=0.8))
		add_model!(dfm, "LDA", LDA())
		add_model!(dfm, "MCLogit", MCLogit(penalty=:l1, lambda=0.01))
		add_model!(dfm, "Rforest", RForest())
		add_model!(dfm, "SVC", SVC())
	end
end

# ╔═╡ 51d745c9-8c1a-41ef-8ee6-c5e9c35d39fe
md"### 1. Training models using the original signal"

# ╔═╡ b0e3e833-47d6-493e-bb51-940267e6f85d
md"To evaluate the LDB algorithm's performance we first train the models using the original signals as input (i.e., Euclidean coordinates). To evaluate the training loss, we will perform a 5 fold cross validation."

# ╔═╡ fded58ea-e7b1-4be1-b952-b7aa1358d5dd
function evaluate_model!(dfm::DataFrame, 
						 df::DataFrame, 
					     model::String, 
						 dat::String, 
						 X, y)
	name = model * "-" * dat
	
	# Training error
	train, test = partition(eachindex(X.train[dat][:,1]), 0.7, shuffle=true)
	M = first(dfm[dfm.name.==model,:model])
	predtype = first(dfm[dfm.name.==model,:predtype])
	
	mach₀ = machine(M, 
					X.train[dat][train,:], 
					y.train["STD"][train])
	MLJ.fit!(mach₀)
	if predtype == :deterministic
		ŷ₀ = predict(mach₀, X.train[dat][test,:])
	else
		ŷ₀ = predict_mode(mach₀, X.train[dat][test,:])
	end
		
	trainacc = Accuracy()(ŷ₀, y.train["STD"][test])
	
	mach = machine(M, X.train[dat], y.train["STD"])
	MLJ.fit!(mach)
	
	# Test error
	if predtype == :deterministic
		ŷ = predict(mach, X.test[dat])
	else
		ŷ = predict_mode(mach, X.test[dat])
	end
	
	testacc = Accuracy()(ŷ, y.test["STD"])
	
	push!(df, Dict(:Type=>name, :TrainAcc=>trainacc, :TestAcc=>testacc))
end

# ╔═╡ 19e7d3f3-970d-4d05-9664-8fe23009fb71
begin
	if autorun2
		for name in dfm.name
			evaluate_model!(dfm, df, name, "STD", X, y)
		end
	end
end

# ╔═╡ 4ffbeab9-67c5-46a0-8e09-449d91dfa34c
if autorun2
	df
end

# ╔═╡ 97516db4-c019-49a7-b826-64294fd14220
md"### Using LDB-k"

# ╔═╡ c47d64f3-12fc-4628-9162-21980066bd00
md"Next, we significantly reduce the dimensionality of the models by only using the top k most discriminant features obtained from LDB."

# ╔═╡ a828877d-1f49-4b76-b397-869bb11e40c5
begin
	if autorun2
		ldbk = "LDB-" * string(dim2)
		
		ldb = LocalDiscriminantBasis(wt = wavelet(eval(Meta.parse(wavelet_type))),
									 dm = dm[d_measure],
									 en = em[e_measure],
									 n_features = dim2)
		WaveletsExt.fit!(ldb, X_train, y_train)
		X.train[ldbk] = DataFrame(WaveletsExt.transform(ldb, X_train)', :auto);
		X.test[ldbk] = DataFrame(WaveletsExt.transform(ldb, X_test)', :auto);
	end
end;

# ╔═╡ 34ff82ef-e7a7-4df2-ab71-3280a5ef34fe
begin
	if autorun2
		for name in dfm.name
			evaluate_model!(dfm, df, name, ldbk, X, y)
		end
	end
end

# ╔═╡ 407cce96-73cb-4baf-90f9-b46d5d617018
if autorun2
	df
end

# ╔═╡ 7dd079af-0445-436c-9bd3-9550cfaa9b19
md"### 3. Using all LDB features"

# ╔═╡ b31e54a1-f1b7-44c4-b2bc-99123933c289
md"Finally, we use all the LDB features to train our models. Note that we do not include the LDA model because theoretically it is the same with using the euclidean coordinates."

# ╔═╡ a08dd1ea-a403-41b4-915c-56fde82222e7
if autorun2
	ldb₁ = LocalDiscriminantBasis(
		wt = wavelet(eval(Meta.parse(wavelet_type))),
		dm = dm[d_measure2],
		en = em[e_measure2]
	)
end

# ╔═╡ 3a374ac2-e225-4637-9dbd-6644cb80b629
begin
	if autorun2
		WaveletsExt.fit!(ldb₁, X_train, y_train)
		X.train["LDB"] = DataFrame(WaveletsExt.transform(ldb₁, X_train)', :auto)
		X.test["LDB"] = DataFrame(WaveletsExt.transform(ldb₁, X_test)', :auto)
		for machine in dfm.name
			evaluate_model!(dfm, df, machine, "LDB", X, y)
		end
	end
end

# ╔═╡ 3f3d4bcf-3f2b-4140-ba52-2246c5140303
if autorun2
	sort!(df, :TestAcc, rev=true)
end

# ╔═╡ 9b292713-1580-48ae-b9cc-05dca097a673
md"## Results"

# ╔═╡ c1b823e9-4a80-4cca-9527-5b0f2933933d
begin
	if autorun2
		set_default_plot_size(18cm, 16cm)
		Gadfly.plot(
			sort(df, :TestAcc),
			layer(
				x=:TestAcc, y=:Type, 
				Geom.point, 
				color=[colorant"#de425b"],
				size = [1.5mm],
				alpha = [0.7]
			),
			layer(
				x=:TrainAcc, y=:Type, 
				Geom.point, 
				color=[colorant"#488f31"],
				size = [1.5mm],
				alpha = [0.7]
			),
			layer(
				xintercept = [1/3],
				Geom.vline(color = "gray", size = [1mm])
			),
			Guide.title("Model Performance"),
			Guide.xlabel("Accuracy"),
			Guide.manual_color_key(
				"",
				["Test Acc","Train Acc", "Baseline"], 
				[colorant"#de425b", colorant"#488f31", "gray"],
				size = [1.5mm],
				shape = [Gadfly.Shape.circle, Gadfly.Shape.circle, Gadfly.Shape.square]
			)
		)
	end
end

# ╔═╡ e58b5f6a-c4da-4125-be5a-7f9125d3f326


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Wavelets = "29a6e085-ba6d-5f35-a997-948ac2efa89a"
WaveletsExt = "8f464e1e-25db-479f-b0a5-b7680379e03f"

[compat]
DataFrames = "~1.3.4"
Distributions = "~0.25.65"
Gadfly = "~1.3.4"
MLJ = "~0.18.4"
Plots = "~1.31.2"
PlutoUI = "~0.7.39"
Wavelets = "~0.9.4"
WaveletsExt = "~0.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "6ccb71b40b04ad69152f1f83d5925de13911417e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.19"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "7d255eb1d2e409335835dc8624c35d97453011eb"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.14"

[[deps.ArrayInterfaceOffsetArrays]]
deps = ["ArrayInterface", "OffsetArrays", "Static"]
git-tree-sha1 = "c49f6bad95a30defff7c637731f00934c7289c50"
uuid = "015c0d05-e682-4f19-8f0a-679ce4c54826"
version = "0.1.6"

[[deps.ArrayInterfaceStaticArrays]]
deps = ["Adapt", "ArrayInterface", "ArrayInterfaceStaticArraysCore", "LinearAlgebra", "Static", "StaticArrays"]
git-tree-sha1 = "efb000a9f643f018d5154e56814e338b5746c560"
uuid = "b0d46f97-bff5-4637-a19a-dd75974142cd"
version = "0.1.4"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AverageShiftedHistograms]]
deps = ["LinearAlgebra", "RecipesBase", "Statistics", "StatsBase", "UnicodePlots"]
git-tree-sha1 = "8bdad2055f64dd71a25826d752e0222726f25f20"
uuid = "77b51b56-6f8f-5c3a-9cb4-d71f9594ea6e"
version = "0.8.7"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "eaee37f76339077f86679787a71990c4e465477f"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.4"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "b1a532a582dd18b34543366322d390e1560d40a9"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.23"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5f5a975d996026a8dd877c35fe26a7b8179c02ba"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.6"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes", "UnicodePlots"]
git-tree-sha1 = "8b35ae165075f95415b15ff6f3c31e879affe977"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.7"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "ff38036fb7edc903de4e79f32067d8497508616b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.2"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "5522c338564580adf5d58d91e43a55db0fa5fb39"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.10"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "3fb5d9183b38fdee997151f723da42fb83d1c6f2"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.6"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "429077fd74119f5ac495857fd51f4120baf36355"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.65"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "037a1ca47e8a5989cc07d19729567bb71bfabd0c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "c8ab731c9127cd931c93221f65d6a1008dad7256"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.0+0"

[[deps.Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "13b402ae74c0558a83c02daa2f3314ddb2d515d3"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "b7b88a4716ac33fe31d6556c02fc60017594343c"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.8"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "d3e45c5307af4b63fd271c9b07634ecb29c737bc"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.1"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "Statistics"]
git-tree-sha1 = "0c703732335a75e683aec7fdfc6d5d1ebd7c596f"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.3"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "d7df9a6fdd82a8cfdfe93a94fcce35515be634da"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.3"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b67e749fb35530979839e7b4b606a97105fe4f1c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.10"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SIMDTypes", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "adc9421494fd93e31a18a66e49d79615ad6b2efa"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.120"

[[deps.LossFunctions]]
deps = ["InteractiveUtils", "Markdown", "RecipesBase"]
git-tree-sha1 = "53cd63a12f06a43eef6f4aafb910ac755c122be7"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.8.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "4199f3ff372222dbdc8602b70f8eefcd1aa06606"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.18.4"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "Serialization", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "62aae83124850bb81cb58a6eb4dc82a12c257ff6"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.20.12"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "ed2f724be26d0023cade9d59b55da93f528c3f26"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.3.1"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "024d0bd22bf4a5b273f626e89d742a9db95285ef"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.5.0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8291b42d6bf744dda0bfb16b6f0befbae232a1fa"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.15.9"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "e1d0220d8bf5c17270cef41835ed57f88d63579d"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.7.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.MarchingCubes]]
deps = ["StaticArrays"]
git-tree-sha1 = "3bf4baa9df7d1367168ebf60ed02b0379ea91099"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.3"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "891d3b4e8f8415f53108b4918d0183e61e18015b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg"]
git-tree-sha1 = "06080992e86a93957bfe2e12d3181443cedf2400"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.2.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "b29873144e57f9fcf8d41d107138a4378e035298"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "4cd738fca4d826bef1a87cbe43196b34fa205e6d"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.6"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "d6de04fd2559ecab7e9a683c59dcbc7dbd20581a"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.1.5"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "4be53d093e9e37772cc89e1009e8f6ad10c4681b"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "b327e4db3f2202a4efafe7569fcbe409106a1f75"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2690681814016887462cf5ac37102b51cd9ec781"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "3177100077c68060d63dd71aec209373c3ec339b"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "dd4195d308df24f33fb10dde7c22103ba88887fa"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.1"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "7ee0e13ac7cd77f2c0e93bff8c40c45f05c77a5a"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.33"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "ba70c9a6e4c81cc3634e3e80bb8163ab5ef57eb8"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "46638763d3a25ad7818a15d441e0c3446a10742d"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "e972716025466461a3dc1588d9168334b71aafff"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.1"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "472d044a1c8df2b062b23f222573ad6837a615ba"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.19"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "ae67ab0505b9453655f7d5ea65183a1cd1b3cfa0"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.12.4"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "953ba1475022a4de16439857a8f79831abf5fa30"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.42"

[[deps.Wavelets]]
deps = ["DSP", "FFTW", "LinearAlgebra", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "52e87ecea56e02e0672c7f3c9fd9ca03915d7e1b"
uuid = "29a6e085-ba6d-5f35-a997-948ac2efa89a"
version = "0.9.4"

[[deps.WaveletsExt]]
deps = ["AverageShiftedHistograms", "Combinatorics", "Distributions", "ImageQualityIndexes", "LinearAlgebra", "Parameters", "Plots", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Wavelets"]
git-tree-sha1 = "4cabb020c5d63e1185dda25043c267f5407fe154"
uuid = "8f464e1e-25db-479f-b0a5-b7680379e03f"
version = "0.2.1"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─45f88030-a821-11eb-0c6d-f5c7c82b7115
# ╠═5ad9f0fb-3688-4b15-94c1-a18e5f41eeed
# ╠═aa7ba1e0-ff23-4406-96a1-4406b4e94399
# ╟─f3a892ef-0c5a-4d4a-86ab-036d913d9d97
# ╟─c195f5d9-2538-4278-9d27-c14446e7cb65
# ╟─a751cd87-80c5-48b1-b798-f1aecebc08a1
# ╠═4e5fe030-8a87-4f4a-88a4-b7b824157880
# ╟─b8077eb3-ec64-4a84-9dcc-3aafce015597
# ╟─e8182e69-8776-4ab5-a38e-bf2175ceb0ea
# ╟─910d24a0-8c00-42c5-8e11-13da2557a09d
# ╟─a0ae476e-7939-4bfe-83e4-9666d0ed366e
# ╟─705723ac-b0e0-4205-b3aa-8b516f9233d4
# ╟─dc92cbff-ccf1-4355-8d60-0e2f39dac6b0
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
# ╟─fb91da71-303f-4c43-be7b-e39df1429355
# ╟─af1b235e-6fff-478f-a5c1-38fbc6c39b8f
# ╟─d27634a5-e703-4fa2-bc1a-d7297b2388a3
# ╟─e0a525ed-35f0-48cc-8403-ddfe03871074
# ╟─9eee7238-6e9c-4837-a30b-ebd09abdcca6
# ╟─fd63c142-ae62-40a2-b34f-986c803ddb72
# ╟─4f8a7bb5-db64-4f82-8544-c961cca068db
# ╟─a0525192-f1d9-4173-960e-ea3c009e067b
# ╟─e77667ca-9bb8-4f30-b5ba-ff107eb9a568
# ╟─885cc8dd-dc5a-4d28-be72-2e26613ec252
# ╟─ed92e98f-e823-45a6-a401-342f584c333e
# ╟─0b12ee12-9229-486f-aa65-1da5c53955cc
# ╟─05a9e8db-fce0-4d12-b67b-c0089621ae07
# ╟─28604f68-a957-4a3c-92f5-13a0ff4ba158
# ╟─b27a4714-cbda-417e-85e1-26d7d98780ee
# ╟─7dc1ba75-fa54-4d59-9b54-93e01da7211e
# ╟─86df0e34-8f77-4d6e-8f40-6f4d8e706c15
# ╟─ad3fe2dc-8003-451c-bf83-c3c7f24e7f0b
# ╟─f0c4b67e-208e-41a4-9510-c47e04a65e20
# ╟─9c4cf3a1-cd6a-4a42-acce-ceaca6c66df2
# ╟─299c91a6-c315-4299-97fc-57a6b6f419b5
# ╟─df7c5ef9-73ff-44b7-aacb-d5fa132d7c2b
# ╟─f1c6268b-a6d5-445a-9e52-748898ec08da
# ╟─9e523918-daaf-4c17-851a-7fac12b04cd3
# ╟─22a8dfdb-c046-4614-8e2b-aab22d12b205
# ╟─09bc8c83-2f25-44a0-81ab-9f0a5724673f
# ╠═2c9b5aef-ba63-44b6-8ef9-ca31cc682fad
# ╟─ae624404-0770-41e9-962b-139006356ea6
# ╟─f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
# ╟─f1727ea3-7353-4c6a-bc13-51622c615780
# ╟─82ffc65d-54ea-42ae-a7ef-dbe6216b0d1e
# ╟─121fd299-5e82-4159-8472-5d385c736c18
# ╟─96a49a0c-4646-43f9-98c2-09ac484f6df8
# ╟─406e7ffe-fa01-4622-ae09-aca5473bfe6c
# ╟─a5126ad3-19b1-4b4e-b96f-ef6b5220854b
# ╟─f9a60263-7ebd-4df8-b33f-5f0e85719186
# ╟─6fef2648-058a-4136-8108-38c1624a19ca
# ╟─8f8bb837-ed86-4a75-8254-913530ed8bc5
# ╟─2999257a-03bf-4313-8dd6-d2da0ed2ef9c
# ╟─65d45fbd-09bf-49e9-b027-e43751ce070f
# ╟─bf8314d6-eb38-4594-afb0-eed5f3151389
# ╟─dde5cc7d-1840-49a9-bcd0-bf3ed6e66007
# ╟─1b71478b-a386-416b-97dc-a2e5da1ce071
# ╟─2bf768ee-7bc7-4039-85d2-dbdf9ed1f75a
# ╟─60cffbcf-d539-4038-9a12-c40fa41d6880
# ╠═7a9548a4-c528-41af-bba7-a99b0c91247b
# ╠═4774bfcf-9e50-428c-b51f-76a887031862
# ╟─2d398c73-37bc-44d4-8559-e220de94624d
# ╠═7a7cae84-3272-4303-80fa-d56a8615b9ff
# ╟─54fdabe6-85ff-4928-ac1c-1555d89ce456
# ╠═9ddb4726-5a78-4adf-a3eb-0796636467c1
# ╟─51d745c9-8c1a-41ef-8ee6-c5e9c35d39fe
# ╟─b0e3e833-47d6-493e-bb51-940267e6f85d
# ╠═fded58ea-e7b1-4be1-b952-b7aa1358d5dd
# ╠═19e7d3f3-970d-4d05-9664-8fe23009fb71
# ╟─4ffbeab9-67c5-46a0-8e09-449d91dfa34c
# ╟─97516db4-c019-49a7-b826-64294fd14220
# ╟─c47d64f3-12fc-4628-9162-21980066bd00
# ╠═a828877d-1f49-4b76-b397-869bb11e40c5
# ╠═34ff82ef-e7a7-4df2-ab71-3280a5ef34fe
# ╠═407cce96-73cb-4baf-90f9-b46d5d617018
# ╟─7dd079af-0445-436c-9bd3-9550cfaa9b19
# ╟─b31e54a1-f1b7-44c4-b2bc-99123933c289
# ╠═a08dd1ea-a403-41b4-915c-56fde82222e7
# ╠═3a374ac2-e225-4637-9dbd-6644cb80b629
# ╠═3f3d4bcf-3f2b-4140-ba52-2246c5140303
# ╟─9b292713-1580-48ae-b9cc-05dca097a673
# ╠═c1b823e9-4a80-4cca-9527-5b0f2933933d
# ╠═e58b5f6a-c4da-4125-be5a-7f9125d3f326
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
