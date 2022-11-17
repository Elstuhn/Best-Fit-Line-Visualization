include("./funcs.jl")
using PlotlyJS, Random


points = 100
epochs = 50
lr = 0.01
m = 0
c = 0

xs = randn(100) .+ 5
ys = randn(100) .+ 5

m, c = train(xs, ys, epoch, lr, m, c)
visualize(xs, ys, m, c)
f(x) = m*x+c
value = rand(range(1, 20))
pred = f(value)
