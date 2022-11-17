using PlotlyJS, Random

points = 100
epochs = 50
lr = 0.01
m = 0
c = 0

xs = randn(100) .+ 5
ys = randn(100) .+ 5

for i in 1:epochs 
    global m
    global c
    y_pred = m .*  xs .+ c 
    D_m = (-2/points) * sum(xs .* (ys - y_pred))
    D_c = (-2/points) * sum(ys - y_pred)
    m = m - lr*D_m 
    c = c - lr*D_c 

end
eqn(x) = m*x+c 
xline = [1:1:points;]
yline = eqn.(xline)

plot([
    scatter(x=xs, y=ys, mode="markers", name="points"),
    scatter(x=xline, y=yline, mode="lines", name="bestfitLine")
])
