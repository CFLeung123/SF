# Define the input text
input_text = """
p_11(L=4)=  -3.67684393132183026620719243289336232e-02

p_11(L=5)=  -4.04690492227564739193475988707781569e-02

p_11(L=6)=  -4.32546872202945850700520235181465018e-02
 
p_11(L=7)=  -4.54837687910094610405390012035812285e-02

p_11(L=8)=  -4.73204219966916208184523421283411767e-02

p_11(L=9)=  -4.88690023374854247173659460163522283e-02

p_11(L=10)=  -5.02011261294437053598022286254192741e-02

p_11(L=11)=  -5.13667229991288805468409586994323497e-02

p_11(L=12)=  -5.24011358527335769010442890080558188e-02

p_11(L=13)=  -5.33299190482535744856687589374292013e-02

p_11(L=14)=  -5.41719976514678789479329899395375618e-02

p_11(L=15)=  -5.4941721072371787243047620454529018e-02

p_11(L=16)=  -5.56502056797480250101671048597553023e-02

p_11(L=17)=  -5.63062279845386098457471512264813632e-02

p_11(L=18)=  -5.69168325121056955836494454011391975e-02

p_11(L=19)=  -5.7487755531087634843552439428919385e-02

p_11(L=20)=  -5.80237272684419083160503017350748174e-02

p_11(L=21)=  -5.85286920147215342122396660229159209e-02

p_11(L=22)=  -5.90059714555819010135542849328897139e-02

p_11(L=23)=  -5.94583879056578023383035481861349777e-02

p_11(L=24)=  -5.98883586750009223940408863968121273e-02

p_11(L=25)=  -6.0297969293613736565855535605404799e-02

p_11(L=26)=  -6.06890310125810315796998319515201808e-02

p_11(L=27)=  -6.10631264490538874951371045917573995e-02

p_11(L=28)=  -6.14216461788415123503254368447444002e-02

p_11(L=29)=  -6.17658183383307785098995053827473794e-02

p_11(L=30)=  -6.20967327714270396520179985573787055e-02

p_11(L=31)=  -6.24153608789079172364021407460063396e-02

p_11(L=32)=  -6.27225720519303693739354036955504591e-02

p_11(L=33)=  -6.30191473681517703818455038530250049e-02

p_11(L=34)=  -6.33057910773509086748690058587621345e-02

p_11(L=35)=  -6.35831402892516234009775824703086511e-02

p_11(L=36)=  -6.38517731894146797614014154570642705e-02

p_11(L=37)=  -6.41122160424350821935196997622073866e-02

p_11(L=38)=  -6.43649491901334548981524461455100622e-02

p_11(L=39)=  -6.46104122122367186999114468420879482e-02

p_11(L=40)=  -6.48490083854734257400465709942626457e-02

p_11(L=41)=  -6.5081108552040871780360567606729727e-02

p_11(L=42)=  -6.53070544885248947514211415620421236e-02

p_11(L=43)=  -6.55271618504327610168396397149662151e-02

p_11(L=44)=  -6.57417227546727680177935000251594011e-02

p_11(L=45)=  -6.59510080519225151623365439925529817e-02

p_11(L=46)=  -6.61552693323644526417013321692752894e-02

p_11(L=47)=  -6.63547407013398371727388829502617308e-02

p_11(L=48)=  -6.65496403557747419242697917061086817e-02

p_11(L=49)=  -6.67401719875244994079273342093460084e-02

p_11(L=50)=  -6.69265260358768492289192962614668771e-02

p_11(L=51)=  -6.71088808081992570693332206563147216e-02

p_11(L=52)=  -6.72874034849929582260220857796097164e-02

p_11(L=53)=  -6.74622510233296416876242261207606359e-02

p_11(L=54)=  -6.76335709707194005821440830465045367e-02

p_11(L=55)=  -6.7801502199828457462907013113242202e-02

p_11(L=56)=  -6.7966175573081759133192010653460822e-02

p_11(L=57)=  -6.81277145450076862711532790750135124e-02

p_11(L=58)=  -6.8286235709176168923475166572700275e-02

p_11(L=59)=  -6.84418492957197779729647347519139608e-02

p_11(L=60)=  -6.85946596246870635935118652832169169e-02

p_11(L=61)=  -6.87447655198396557124460199704526761e-02

p_11(L=62)=  -6.8892260686953750434239858604189351e-02

p_11(L=63)=  -6.90372340602095166439607953260846078e-02

p_11(L=64)=  -6.91797701198377485347864655526164588e-02
"""



#Extrapolation
using DoubleFloats
using Quadmath
using Plots
const delta = 1
const Lmin = 4

# Extract the numbers using a regular expression
f = [parse(Float128, match(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", line).match) for line in split(input_text, '\n') if match(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", line)!==nothing]

print(f)

function R0(f)
    R0f = Array{Float128}(undef, length(f) - delta)
    for i in 1:length(f)-delta
        L = i - 1 + Lmin
        R0f[i] = Float128(L / delta) * (f[i+delta] - f[i])
    end
    return R0f
end

function Rnu(nu, f)
    Rnuf = Array{Float128}(undef, length(f) - delta)
    for i in 1:length(f)-delta
        L = i - 1 + Lmin
        Rnuf[i] = f[i] + Float128(L / (nu * delta)) * (f[i+delta] - f[i])
    end
    return Rnuf
end

# Recursive extrapolation function
function recursive_extrapolation(data, nu, max_nu, depth)
    if depth > 2
        nu += 1
        depth = 1
    end
    
    if nu > max_nu
        return data
    end
    
    new_data = Rnu(nu, data)
    
    # Apply Rnu twice recursively
    result = recursive_extrapolation(new_data, nu, max_nu, depth + 1)
    
    # Plot after every second Rnu application (when depth is even)
    if depth == 2
        L_values = []
        f_values = []
        for l in 1:length(new_data)
            L = l - 1 + Lmin
            if L > 32
                push!(L_values, L)
                push!(f_values, new_data[l])
            end
        end
        
        if !isempty(L_values)
            plot!(L_values, f_values, label="nu=$nu", markershape=:auto, markersize=3)
        end
    end
    
    return result
end

# Create plot for intermediate steps
plot(legend=:bottomright, xlabel="L", ylabel="s0", title="Extrapolation Steps", grid=true, dpi=300)


# Start extrapolation
data0 = R0(f)
final = recursive_extrapolation(data0, 1, 7,1)


# Add initial R0 data for L>32
L0_values = []
f0_values = []
for l in 1:length(data0)
    L = l - 1 + Lmin
    if L > 32
        push!(L0_values, L)
        push!(f0_values, data0[l])
    end
end
scatter!(L0_values, f0_values, label="R0", markersize=3)

# Add baseline and final result
L_final_values = []
f_final_values = []
for l in 1:length(final)
    L = l - 1 + Lmin
    if L > 32
        push!(L_final_values, L)
        push!(f_final_values, final[l])
    end
end
scatter!(L_final_values, f_final_values, label="Final", markersize=4, color=:black)

hline!([-0.0084434319], label="baseline=-0.0084434319", linestyle=:dash, color=:red)

# Set axis limits
xlims!(minimum(L_final_values), 20+maximum(L_final_values))
ylims!(-0.00844350, -0.00844340)

# Save and display
savefig("extrapolation_steps.png")
display(plot!())

# Print final results and errors
println("Final Extrapolation Results:")
for l in 1:length(final)
    L = l - 1 + Lmin
    if L > 32
        result = final[l]
        abs_error = result + 1/(12 * Float128(pi)^2)
        percent_error = 100 * (result * (-12 * Float128(pi)^2) - 1)
        println("L=$L: Result=$result, Abs Error=$abs_error, %Error=$percent_error")
    end
end
println(" -------------------------------------------------------- ")
