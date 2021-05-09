
h₁(i::Int) = max(6 - abs(i-7), 0)
h₂(i::Int) = h₁(i - 8)
h₃(i::Int) = h₁(i - 4)

"""
  triangular_test_function(c1::Int, c2::Int, c3::Int, L::Int=32)

Generates a set of triangluar test functions with 3 classes.
"""
function triangular_test_functions(c1::Int, c2::Int, c3::Int; L::Int=32, shuffle::Bool=false)
  @assert c1 >= 0
  @assert c2 >= 0
  @assert c3 >= 0

  u = rand(Uniform(0,1),1)[1]
  ϵ = rand(Normal(0,1),(L,c1+c2+c3))

  y = string.(vcat(ones(c1), ones(c2) .+ 1, ones(c3) .+ 2))

  H₁ = Array{Float64,2}(undef,L,c1)
  H₂ = Array{Float64,2}(undef,L,c2)
  H₃ = Array{Float64,2}(undef,L,c3)
  for i in 1:L
    H₁[i,:] .= u * h₁(i) + (1 - u) * h₂(i)
    H₂[i,:] .= u * h₁(i) + (1 - u) * h₃(i)
    H₃[i,:] .= u * h₂(i) + (1 - u) * h₃(i)
  end

  H = hcat(H₁, H₂, H₃) + ϵ

  if shuffle
    idx = [1:(c1+c2+c3)...]
    shuffle!(idx)
    return H[:,idx], y[idx]
  end

  return H, y
end