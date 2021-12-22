module Datasets2D

export textures_label, get_dataset

import HTTP, Tar
import CodecZlib: GzipDecompressorStream
import BufferedStreams: BufferedInputStream
import FileIO: load
import StatsBase: sample
import WaveletsExt: ClassData, generateclassdata
using MLDatasets

include("textures_dict.jl")
const textures_url = "https://sipi.usc.edu/database/textures.tar.gz"

function extract_tgz_from_url(link::String, dir::String = "./data/")
    r = HTTP.request(:GET, link)
    Tar.extract(GzipDecompressorStream(BufferedInputStream(r.body)), dir)
end

"""
    get_textures_dataset(list, train_size, test_size[, dir])

# Arguments
- `list::AbstractVector{T} where T<:AbstractString`
- `train_size::Union{S, AbstractVector{S}} where S<:Integer`
- `test_size::Union{S, AbstractVector{S}} where S<:Integer`
- `dir::T where T<:AbstractString`: (Default: `"./data/textures/`)

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

function get_mnist_dataset(dir::String = "./data/MNIST/", args...)
    # Downloads the MNIST dataset
    isdir(dir) || MNIST.download(dir, i_accept_the_terms_of_use = true)
    # Unpack the downloaded data into variables
    train_x, train_y = MNIST.traindata(dir = dir)
    test_x, test_y = MNIST.testdata(dir = dir)
    # Zero padding to make 28x28 images into 32x32 (using default settings)
    train_x = cat(train_x, dims = 3) |> x -> zero_padding(x, args...)
    test_x = cat(test_x, dims = 3) |> x -> zero_padding(x, args...)
    return (train_x, train_y), (test_x, test_y)
end

function zero_padding(data::AbstractArray{T,3}, pad_widths::Integer = 2) where T
    m, n, k = size(data)
    padded = zeros(T, (m+2*pad_widths, n+2*pad_widths, k))
    padded[(pad_widths+1):(end-pad_widths), (pad_widths+1):(end-pad_widths), :] = data
    return padded
end

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

function get_dataset(dataset::Symbol, args...)
    @assert dataset ∈ [:mnist, :textures, :tri, :cbf]
    return @eval $(Symbol("get_$(dataset)_dataset"))($args...)
end

end