using PlotlyJS

function train(x::Vector{Float64}, y::Vector{Float64}, epoch::Int64, lr::Float64, m = 0, c = 0)
  #=
  x and y accepts Vector{Float64}, Vector{Float32}, Vector{Int64}, Vector{Int32}
  returns m and c values
  =#
  n = length(x)
  if n != length(y)
    error("length of x and y has to be the same")
  elseif !Bool(n)
    error("size of datasets cannot be 0")
  end
  for i in 1:epochs 
    global m
    global c
    y_pred = m .*  x .+ c 
    d_m = (-2/n) * sum(x .* (y - y_pred))
    d_c = (-2/n) * sum(y - y_pred)
    m = m - lr*d_m
    c = c - lr*d_c
  end
  return (m, c)
end


function visualize(x::Vector{Float64}, y::Vector{Float64}, m::Float64, c::Float64)
  #=
  x and y accepts Vector{Float64}, Vector{Float32}, Vector{Int64}, Vector{Int32}
  x and y should be training data and both m and c values should correspond to training data
  =#
  n = length(x)
  if n != length(y)
    error("length of x and y has to be the same")
  elseif !Bool(n)
    error("size of datasets cannot be 0")
  end
  f(x) = m*x+c
  xs = [1:1:n;]
  ys = f.(xs)
  plot([
    scatter(x=x, y=y, mode="markers", name="points"),
    scatter(x=xs, y=ys, mode="lines", name="bestfitline")
  ])
end
