module LDBDatasets

export textures_label, 
       get_textures_dataset,
       get_mnist_dataset,
       get_1d_dataset,
       get_dataset

import HTTP, Tar
import CodecZlib: GzipDecompressorStream
import BufferedStreams: BufferedInputStream
import FileIO: load
import StatsBase: sample
import WaveletsExt: ClassData, generateclassdata
using MLDatasets

include("textures_dict.jl")
const textures_url = "https://sipi.usc.edu/database/textures.tar.gz"

"""
    extract_tgz_from_url(link, [dir])

Decompress and extract files from a .tar.gz file that is obtained from `link` into the
directory `dir`.

# Argummets
- `link::String`: URL link to download .tar.gz file.
- `dir::String`: (Default: `./data/`) Directory to save decompressed and extracted files.
"""
function extract_tgz_from_url(link::String, dir::String = "./data/")
    r = HTTP.request(:GET, link)
    Tar.extract(GzipDecompressorStream(BufferedInputStream(r.body)), dir)
end

"""
    get_textures_dataset(list, train_size, test_size, [dir])

Obtains the textures dataset.

# Arguments
- `list::AbstractVector{T} where T<:AbstractString`: List of texture files to generate
  classes of data. See the full list of texture files in the `textures_label` variable in
  `textures_dict.jl`.
- `train_size::Union{S, AbstractVector{S}} where S<:Integer`: Train size. If `train_size` is
  given as a scalar, then each class in `list` will have `train_size` number of data.
  Similarly, if `train_size` is a vector, then each entry is the train size to the
  corresponding entry in `list`.
- `test_size::Union{S, AbstractVector{S}} where S<:Integer`: Test size. If `test_size` is
  given as a scalar, then each class in `list` will have `test_size` number of data.
  Similarly, if `test_size` is a vector, then each entry is the test size to the
  corresponding entry in `list`.
- `dir::T where T<:AbstractString`: (Default: `"./data/textures/`) Directory to save
  decompressed and extracted textures files.

# Returns
Output in the form of `(train_x, train_y), (test_x, test_y)`
"""
function get_textures_dataset(list::AbstractVector{T}, 
                              train_size::AbstractVector{S},
                              test_size::AbstractVector{S},
                              dir::T = "./data/textures/") where 
                             {T<:AbstractString, S<:Integer}
    @assert length(list) == length(train_size) == length(test_size)
    # Downloads the textures dataset
    if ~isdir(dir) || "textures" ∉ readdir(dir)
        extract_tgz_from_url(textures_url, dir)
    end
    # Get train and test samples
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for (texture, train_sz, test_sz) in zip(list, train_size, test_size)
        filename = textures_label[texture]
        path = dir * "textures/" * filename
        (tr_x, tr_y), (te_x, te_y) = generate_texture_samples(path, texture, train_sz, test_sz)
        push!(train_x, tr_x)
        push!(train_y, tr_y)
        push!(test_x, te_x)
        push!(test_y, te_y)
    end
    train_x = cat(train_x..., dims = 3)
    train_y = vcat(train_y...)
    test_x = cat(test_x..., dims = 3)
    test_y = vcat(test_y...)
    return (train_x, train_y), (test_x, test_y)
end

function get_textures_dataset(list::AbstractVector{T}, train_size::S, test_size::S, 
    dir::T = "./data/textures/") where
    {T<:AbstractString, S<:Integer}
    # Each class in list will have `train_size` number of training data, and `test_size`
    # number of test data
    n = length(list)
    train_size = repeat([train_size], n)
    test_size = repeat([test_size], n)
    return get_textures_dataset(list, train_size, test_size, dir)
end

"""
    generate_texture_samples(path, texture, train_size, test_size)

Generate samples of size 128 x 128 for class `texture`.

# Arguments
- `path::T where T<:AbstractString`: Path of the `texture` file.
- `texture::T where T<:AbstractString`: The corresponding texture name.
- `train_size::S where S<:Integer`: Train size.
- `test_size::S where S<:Integer`: Test size.

# Returns
Output in the form of `(train_x, train_y), (test_x, test_y)`
"""
function generate_texture_samples(path::T, texture::T, train_size::S, test_size::S) where
                                 {T<:AbstractString, S<:Integer}
    # Load image and convert to elements to Float64
    img = load(path) |> x -> convert.(Float64, x)
    # Size of image
    m, n = size(img)
    # Get Cartesian Indices 
    indices = CartesianIndices((1:(m-127), 1:(n-127)))
    # Randomly sample the cartesian indices
    samples = sample(indices, train_size+test_size, replace = false)
    # Generate subimage
    x = Array{Float64,3}(undef, (128,128,train_size+test_size))
    for (k,s) in enumerate(samples)
        i = s[1]
        j = s[2]
        x[:,:,k] = img[i:(i+127),j:(j+127)]
    end
    # Split to train and test sets
    train_x = x[:,:,1:train_size]
    train_y = repeat([texture], train_size)
    test_x = x[:,:,(train_size+1):end]
    test_y = repeat([texture], test_size)
    return (train_x, train_y), (test_x, test_y)
end

"""
    get_mnist_dataset([dir], [pad_widths])

Obtains the MNIST dataset.

# Arguments
- `dir::T where T<:AbstractString`: (Default: `"./data/textures/`) Directory to save
  decompressed and extracted textures files.
- `pad_widths::Integer`: (Default: `2`) Width of zero padding. Used to reshape the size of
  images into that of power of 2 for full wavelet decomposition capabilities.

# Returns
Output in the form of `(train_x, train_y), (test_x, test_y)`
"""
function get_mnist_dataset(dir::String = "./data/MNIST/", pad_widths::Integer = 2)
    # Downloads the MNIST dataset
    isdir(dir) || MNIST.download(dir, i_accept_the_terms_of_use = true)
    # Unpack the downloaded data into variables
    train_x, train_y = MNIST.traindata(dir = dir)
    test_x, test_y = MNIST.testdata(dir = dir)
    # Zero padding to make 28x28 images into 32x32 (using default settings)
    train_x = cat(train_x, dims = 3) |> x -> zero_padding(x, pad_widths)
    test_x = cat(test_x, dims = 3) |> x -> zero_padding(x, pad_widths)
    return (train_x, train_y), (test_x, test_y)
end

"""
    zero_padding(data, [pad_widths])

Zero-padding on `data` of size ``(m, n, k)`` into size ``(m + 2d, n + 2d, k)`` where ``d``
is the padding width.

# Arguments
- `data::AbstractArray{T,3} where T`: Array of size (m, n, k).
- `pad_widths::Integer`: (Default: `2`) Width of zero padding. Used to reshape the size of
  images into that of power of 2 for full wavelet decomposition capabilities.

# Returns
- `padded::Array{T,3}`: Padded `data`.
"""
function zero_padding(data::AbstractArray{T,3}, pad_widths::Integer = 2) where T
    m, n, k = size(data)
    padded = zeros(T, (m+2*pad_widths, n+2*pad_widths, k))
    padded[(pad_widths+1):(end-pad_widths), (pad_widths+1):(end-pad_widths), :] = data
    return padded
end

"""
    get_1d_dataset(type, train_size, test_size)

Wrapper for `WaveletsExt.generateclassdata` for generating 1D class data.

# Arguments
- `type::Symbol`: The type of class data, `:tri` (for triangular waveforms) or `:cbf` (for
  cylinder-bell-funnel).
- `train_size::Union{S, AbstractVector{S}} where S<:Integer`: Train size. If `train_size` is
  given as a scalar, then each class in `list` will have `train_size` number of data.
  Similarly, if `train_size` is a vector, then each entry is the train size to the
  corresponding entry in `list`.
- `test_size::Union{S, AbstractVector{S}} where S<:Integer`: Test size. If `test_size` is
  given as a scalar, then each class in `list` will have `test_size` number of data.
  Similarly, if `test_size` is a vector, then each entry is the test size to the
  corresponding entry in `list`.

# Returns
Output in the form of `(train_x, train_y), (test_x, test_y)`
"""
function get_1d_dataset(type::Symbol, train_size::T, test_size::T) where T<:Integer
    return get_1d_dataset(type, repeat([train_size], 3), repeat([test_size], 3))
end

function get_1d_dataset(type::Symbol,
                         train_size::AbstractVector{T}, 
                         test_size::AbstractVector{T}) where T<:Integer
    @assert length(train_size) == length(test_size) == 3
    train_x, train_y = generateclassdata(ClassData(type, train_size...))
    test_x, test_y = generateclassdata(ClassData(type, test_size...))
    return (train_x, train_y), (test_x, test_y)
end

get_tri_dataset(args...) = get_1d_dataset(:tri, args...)
get_cbf_dataset(args...) = get_1d_dataset(:cbf, args...)

"""
    get_dataset(dataset, args...)

Collect/generate dataset.

# Arguments
- `dataset::Symbol`: Type of class data. Supported types are:
    - `:mnist`: MNIST dataset (2D).
    - `:textures`: Textures dataset (2D).
    - `:tri`: Triangular waveforms (1D).
    - `:cbf`: Cylinder-bell-funnel dataset (1D).
- `args...`: Additional argument for generating dataset. See:
    - `:mnist`: [`get_mnist_dataset`](@ref).
    - `:textures`: [`get_textures_dataset`](@ref).
    - `:tri`: [`get_1d_dataset`](@ref).
    - `:cbf`: [`get_1d_dataset`](@ref).

# Returns
Output in the form of `(train_x, train_y), (test_x, test_y)`
"""
function get_dataset(dataset::Symbol, args...)
    @assert dataset ∈ [:mnist, :textures, :tri, :cbf]
    return @eval $(Symbol("get_$(dataset)_dataset"))($args...)
end

end # module