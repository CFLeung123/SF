using ForwardDiff

f(x, y) = x^2 * y + y^2

x_val, y_val = 3.0, 2.0 # the point where we compute f(x, y), ∂f/∂x, ∂f/∂y, ∂²f/∂x∂y at

# Construct 4D hyper-dual numbers (strictly following mathematical definitions)
# x_hyper = a0 + ε₁a₁ + ε₂a₂ + ε₁ε₂a₃
# Where:
#   a0 = x_val (primal value)
#   a1 = 1.0 (first-order perturbation coefficient for x)
#   a2 = 0.0 (initial first-order perturbation coefficient for y, set to 0)
#   a3 = 0.0 (initial mixed perturbation coefficient, set to 0)
x_hyper = ForwardDiff.Dual(
    ForwardDiff.Dual(x_val, 1.0),  # a0 + ε₁a₁
    ForwardDiff.Dual(0.0, 0.0)      # ε₂a₂ + ε₁ε₂a₃ (here a2=0, a3=0)
)

# y_hyper follows the same logic, but with perturbation in ε₂ direction
y_hyper = ForwardDiff.Dual(
    ForwardDiff.Dual(y_val, 0.0),  # a0 + ε₁a₁ (here a1=0)
    ForwardDiff.Dual(1.0, 0.0)      # ε₂a₂ + ε₁ε₂a₃ (a2=1.0)
)

# Compute function result
result = f(x_hyper, y_hyper)

# Directly unpack results (following mathematical definition hierarchy)
f_value = ForwardDiff.value(ForwardDiff.value(result))  
df_dx = ForwardDiff.partials(ForwardDiff.value(result))[1]  
df_dy = ForwardDiff.partials(result)[1].value               
d2f_dxdy = ForwardDiff.partials(result)[1].partials[1]      

# Output results
println("f(x, y)    = ", f_value)    # 
println("∂f/∂x     = ", df_dx)       # 
println("∂f/∂y     = ", df_dy)       # 
println("∂²f/∂x∂y = ", d2f_dxdy)    
