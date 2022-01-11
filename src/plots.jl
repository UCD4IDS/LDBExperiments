module LDBPlots

export plot_emap,
       plot_emaps

using Plots, Wavelets, WaveletsExt

function plot_emap(emap::AbstractArray{T,2}, tree::BitVector; 
                   clim::Tuple{T,T} = extrema(emap)) where T<:Real
    start = 0                           # start index
    ny, nx = size(emap)                 # emap dimensions
    emap = emap'

    plt = heatmap(start:(ny+start-1),             # x-axis labels
                  0:(nx-1),                       # y-axis labels
                  emap,                           # heatmap values
                  c = :matter,                    # color gradient
                  background_color = :white,
                  legend = false,
                  clim = clim)                    # color bar limit
    # Add tfbdry lines on top
    plot_tfbdry!(plt, tree, node_color = :transparent, line_color = :black, 
                 background_color = :transparent)
    return plt
end

function plot_emap(emap::AbstractArray{T,3}, tree::BitVector;
                   clim::Tuple{T,T} = extrema(emap)) where T<:Real
    @assert isvalidtree(emap[:,:,1], tree)
    nx, ny, _ = size(emap)
    data = getbasiscoef(emap, tree)             # get basis coefficients
    
    plt = heatmap(0:(ny-1),                     # x-axis labels
                  0:(nx-1),                     # y-axis labels
                  data,                         # heatmap values
                  c = :matter,                  # color gradient
                  background_color = :white,
                  legend = false,
                  clim = clim)                  # color bar limit
    plot_tfbdry2!(plt, tree, line_color = :black, background_color = :transparent)
    return plt
end

function plot_emaps(emap::AbstractArray{T}, tree::BitVector; 
                    clim::Tuple{T,T} = extrema(emap)) where T<:Real
    @assert 3 ≤ ndims(emap) ≤ 4
    test_signal = ndims(emap) == 3 ? emap[:,1,1] : emap[:,:,1,1]
    @assert isvalidtree(test_signal, tree)            # Check if tree is valid for 1D signals
    plts = []                                         # Collect all plots
    # Plot energy map for each class
    for (emapᵢ, i) in zip(eachslice(emap, dims=ndims(emap)), axes(emap,ndims(emap)))
        plt = plot_emap(emapᵢ, tree, clim = clim)
        plot!(plt, ylabel = "Class $i")
        push!(plts, plt)
    end
    plts = plot(plts..., layout = (size(emap,ndims(emap)), 1))  # Display all plots
    return plts
end

end # module