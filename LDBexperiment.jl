### A Pluto.jl notebook ###
# v0.17.3

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

# ╔═╡ aa7ba1e0-ff23-4406-96a1-4406b4e94399
begin
	include("./src/data.jl")
	import .LDBDatasets: get_dataset, textures_label
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

$(@bind autorun CheckBox())"

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
md"`WaveletsExt.jl` provides interfaces to understand the results of LDB. For example we can visualize the energy/probability density maps for each class by accessing the `Γ` attribute in the fitted LDB object. The plots below shows the energy/probability density maps for each class, where darker colors indicate that the sub-band has a higher discriminant power."

# ╔═╡ bf1623dc-865d-4339-ae38-74457d7685c1
md"
### TODO: Figure out how to deal with following plot for 2D and ProbabilityDensity case"

# ╔═╡ f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
begin
	function plot_emap(emap::AbstractArray; legend = true, clim=(0,maximum(emap)))
		
		start = 0
		ncol, nrow = size(emap)
		emap = emap'

		p = Plots.heatmap(
			start:(ncol+start-1), 
			0:(nrow-1), 
			emap, 
			c = :matter, 
			background_color = :black,
			legend = legend,
			clim = clim
		)

		Plots.plot!(p, xlims = (start-0.5, ncol+start-0.5), 
			ylims = (-0.5, nrow-0.5), yticks = 0:nrow)
		# plot horizontal lines
		for j in 0:(nrow-1)
			Plots.plot!(p, (start-0.5):(ncol+start-0.5), (j-0.5)*ones(ncol+1), 
				color = :black, legend = false)
		end

		# plot vertical lines
		for j in 1:(nrow-1)
			for jj in 1:2^(j-1)
				vpos = (ncol/2^j)*(2*jj-1)+start-0.5;
				Plots.plot!(p, vpos*ones(nrow-j+1), j-0.5:nrow-0.5, color = :black)
			end
		end
		Plots.plot!(p, (ncol+start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = :black)
		Plots.plot!(p, (start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = :black)
		Plots.plot!(p, yaxis = :flip)

		return p
	end
	if autorun
		gr(size=(700,600))
		hmap1 = plot_emap(ldb₀.Γ[:,:,1], clim = (0,0.6))
		Plots.plot!(ylabel = "Class 1")
		hmap2 = plot_emap(ldb₀.Γ[:,:,2], clim = (0,0.6))
		Plots.plot!(ylabel = "Class 2")
		hmap3 = plot_emap(ldb₀.Γ[:,:,3], clim = (0,0.6))
		Plots.plot!(ylabel = "Class 3")
		Plots.plot(hmap1, hmap2, hmap3, layout=(3,1))
	end
end

# ╔═╡ 82ffc65d-54ea-42ae-a7ef-dbe6216b0d1e
md"We can also visualize the discriminant measure map and the selected nodes/sub-bands using the `discriminant_measure` function and the `tree` attributes respectively. The first plot in the figure below displays the heatmap for the discriminant measure map and the second plot displays the selected nodes from the binary tree."

# ╔═╡ 121fd299-5e82-4159-8472-5d385c736c18
begin
	if autorun
		gr(size=(700,400))
		dmap = plot_emap(
			discriminant_measure(ldb₀.Γ, dm[d_measure]);
			legend = true
		)
		tmap = plot_tfbdry(ldb₀.tree)
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

# ╔═╡ 6fef2648-058a-4136-8108-38c1624a19ca
@bind dim2 Slider(1:length(X₀[:,1]), default=10, show_value=true)

# ╔═╡ 60cffbcf-d539-4038-9a12-c40fa41d6880
md"**Auto Run**: Check the box after you are satisfied with the experiment parameters or when you want to re-run the experiment (uncheck and check again)

$(@bind autorun2 CheckBox())"

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
		X.train["STD"]= reshape(X_train,:,33*nclass_train)'
		X.test["STD"] = reshape(X_test,:,333*nclass_test)'
		y.train["STD"], y.test["STD"] = coerce(y_train, Multiclass), coerce(y_test, Multiclass)
	end
end;

# ╔═╡ 2d398c73-37bc-44d4-8559-e220de94624d
md"Next, we will load some machine learning models from `MLJ.jl`. We will include two very basic decision tree models(with and without pruning), Linear Discriminant classifier (LDA), Multinomial classifier with L1 regularization (i.e., LASSO), and finally a Random Forest classifier." 

# ╔═╡ 7a7cae84-3272-4303-80fa-d56a8615b9ff
begin
	Tree = @load DecisionTreeClassifier pkg=DecisionTree
	LDA = @load LDA pkg=MultivariateStats
	MCLogit = @load MultinomialClassifier pkg=MLJLinearModels
	RForest = @load RandomForestClassifier pkg=DecisionTree
	SVC = @load SVC pkg=LIBSVM
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
		X.train[ldbk] = WaveletsExt.transform(ldb, X_train)';
		X.test[ldbk] = WaveletsExt.transform(ldb, X_test)';
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
		X.train["LDB"] = WaveletsExt.transform(ldb₁, X_train)'
		X.test["LDB"] = WaveletsExt.transform(ldb₁, X_test)'
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

# ╔═╡ Cell order:
# ╟─45f88030-a821-11eb-0c6d-f5c7c82b7115
# ╠═45468d3a-3456-4e99-aec8-b3c41b20a426
# ╠═5ad9f0fb-3688-4b15-94c1-a18e5f41eeed
# ╠═aa7ba1e0-ff23-4406-96a1-4406b4e94399
# ╟─f3a892ef-0c5a-4d4a-86ab-036d913d9d97
# ╟─c195f5d9-2538-4278-9d27-c14446e7cb65
# ╟─a751cd87-80c5-48b1-b798-f1aecebc08a1
# ╟─4e5fe030-8a87-4f4a-88a4-b7b824157880
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
# ╟─bf1623dc-865d-4339-ae38-74457d7685c1
# ╟─f7669f3f-9e34-4dc1-b3d4-7eda7fe6e479
# ╟─82ffc65d-54ea-42ae-a7ef-dbe6216b0d1e
# ╟─121fd299-5e82-4159-8472-5d385c736c18
# ╟─96a49a0c-4646-43f9-98c2-09ac484f6df8
# ╟─406e7ffe-fa01-4622-ae09-aca5473bfe6c
# ╟─a5126ad3-19b1-4b4e-b96f-ef6b5220854b
# ╟─f9a60263-7ebd-4df8-b33f-5f0e85719186
# ╟─8f8bb837-ed86-4a75-8254-913530ed8bc5
# ╟─2999257a-03bf-4313-8dd6-d2da0ed2ef9c
# ╟─65d45fbd-09bf-49e9-b027-e43751ce070f
# ╟─bf8314d6-eb38-4594-afb0-eed5f3151389
# ╟─dde5cc7d-1840-49a9-bcd0-bf3ed6e66007
# ╟─1b71478b-a386-416b-97dc-a2e5da1ce071
# ╟─2bf768ee-7bc7-4039-85d2-dbdf9ed1f75a
# ╟─6fef2648-058a-4136-8108-38c1624a19ca
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
