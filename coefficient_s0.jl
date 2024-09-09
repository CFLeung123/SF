#Extrapolation
using DoubleFloats
using Quadmath
const delta = 1
const Lmin = 4

# Use pwd() in REPL to check the current directory and make sure the file exist
# Define the input text
inputf = open("input.txt","r")
lines = readlines(inputf)
print(lines)
# Extract the numbers using a regular expression
f = [parse(Double64, match(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", line).match) for line in lines if !isempty(line)]

close(inputf)

function R0(f)
    R0f = Array{Float128}(undef, length(f) - delta)
    for L in 1:length(f)-delta
        R0f[L] = Float128(L / delta) * (f[L+delta] - f[L])
    end
    return R0f
end

function Rnu(nu, f)
    Rnuf = Array{Float128}(undef, length(f) - delta)
    for L in 1:length(f)-delta
        Rnuf[L] = f[L] + Float128(L / (nu * delta)) * (f[L+delta] - f[L])
    end
    return Rnuf
end

function R2R2R1R1R0f(f)
    final = Rnu(2, Rnu(2, Rnu(1, Rnu(1, R0(f)))))

    for l in 1:length(final)
        @inbounds begin
            L = l - 1 + Lmin
            println(' ')
            println("Final(L=$L)= ", final[l])
            println("Abs error=  ", final[l] + 1 / (12 * Float128(pi)^2))
            println("%error=  ", 100 * (final[l] * (-12 * Float128(pi)^2) - 1))
        end
    end
end

R2R2R1R1R0f(f)
println(" -------------------------------------------------------- ")
#R2R2R1R1R0f(f2)
