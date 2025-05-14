#using Pkg; Pkg.add("SparseArrays")
#using SparseArrays
using LinearAlgebra
using Printf
using BenchmarkTools
using DoubleFloats
using StaticArrays

BLAS.set_num_threads(14)  # Set the number of threads 


using SymmetricTensors

# Define the parameters
eta = 0
const nu = 0
Lmin = 16
Lmax = 48
m = 0
const theta = Float64(pi) / 5
const c_sw = 1
const rho = 1
#fine structure const alpha = 0 #QED off
const alpha_base = Float64(0.0072973525693)
const Q = Float64(-1 / 3)   #uptype Q=+2/3 downtype Q=-1/3
# photon = \varphi * Q * \frac{e}{a}
const phiQED = Float64(-pi/4)
const phipQED = Float64(-3pi/4)
const m_e = Float64(0.511) # mass of eletron
# set maximum energy scale as 100 MeV
const scale_en = 10000
alpha_max = alpha_base / (1- alpha_base/(3*Float64(pi)) *2*log(scale_en/m_e))
n = 10 # number of data points exempt 0

println("eta=$eta   nu=$nu   Lmin=$Lmin   Lmax=$Lmax   m=$m   theta=$theta   c_sw=$c_sw   L(space)= $rho *L(time) ")



# Define gamma matrices

const gamma0 = SMatrix{4,4}(Diagonal([1, 1, -1, -1])) # gamma0
const gamma = [SMatrix{4,4}([0 0 0 -im; 0 0 -im 0; 0 im 0 0; im 0 0 0]),  # gamma1
    SMatrix{4,4}([0 0 0 1; 0 0 -1 0; 0 -1 0 0; 1 0 0 0]),  # gamma2
    SMatrix{4,4}([0 0 im 0; 0 0 0 -im; -im 0 0 0; 0 im 0 0]) # gamma3
]
const gamma5 = SMatrix{4,4}([0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]) # gamma5

# Define 4x4 identity matrix 
const id = SMatrix{4,4}(Diagonal([1, 1, 1, 1]))
# Define P_+ and P_- constants
const P_plus = SMatrix{4,4}(Int64.(0.5 * (id + gamma0)))
const P_minus = SMatrix{4,4}(Int64.(0.5 * (id - gamma0)))


# Define background fields
function compute_phi(n_c::Int, eta, coup_e)
    if n_c == 1
        phi = eta - Float64(pi) / 3 + phiQED * coup_e * Q
    elseif n_c == 2
        phi = eta * (nu - 0.5) + phiQED * coup_e * Q
    elseif n_c == 3
        phi = -eta * (nu + 0.5) + Float64(pi) / 3 + phiQED * coup_e * Q
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phi
end

function compute_phiprime(n_c::Int, eta, coup_e)
    if n_c == 1
        phip = -(eta - Float64(pi) / 3) - Float64(pi) * 4 / 3 + phipQED * coup_e * Q
    elseif n_c == 2
        phip = -(-eta * (nu + 0.5) + Float64(pi) / 3) + Float64(pi) * 2 / 3 + phipQED * coup_e * Q
    elseif n_c == 3
        phip = -(eta * (nu - 0.5)) + Float64(pi) * 2 / 3 + phipQED * coup_e * Q
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phip
end

function compute_detaphi(n_c::Int, eta)
    if n_c == 1
        detaphi = Float64(1)
    elseif n_c == 2
        detaphi = nu - Float64(1) / Float64(2)
    elseif n_c == 3
        detaphi = -(nu + Float64(1) / Float64(2))
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return detaphi
end

function compute_detaphiprime(n_c::Int, eta)
    if n_c == 1
        detaphip = -Float64(1)
    elseif n_c == 2
        detaphip = nu + Float64(1) / Float64(2)
    elseif n_c == 3
        detaphip = -(nu - Float64(1) / Float64(2))
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return detaphip
end
#=
# Function to calculate eta derivatives
const detaphis = [ Float64(1), nu - Float64(1) / Float64(2), -(nu + Float64(1) / Float64(2))]
const detaphis_prime = [-Float64(1), nu + Float64(1) / Float64(2), -(nu - Float64(1) / Float64(2))]
=#



#print complex matrix with n significant figures
function print_mat(matrix, n)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            real_part = round(real(matrix[i, j]), sigdigits=n)
            imag_part = round(imag(matrix[i, j]), sigdigits=n)
            if (imag_part != 0) && (real_part != 0)
                @printf("%.*g + %.*gi, ", n, real_part, n, imag_part)
            elseif (imag_part != 0)
                @printf("%.*gi, ", n, imag_part)
            else
                @printf("%.*g, ", n, real_part)
            end
        end
        println(" ")
    end
end





# matrix B and B' with fixed time , spatial momentum and color(background fields).
function Calculate_BandB_prime(L::Int64, t::Int64, p::Array{Float64}, n_c::Int64, coup_e)
    # Initialize color related parameters
    phi = compute_phi(n_c, eta, coup_e)
    phi_prime = compute_phiprime(n_c, eta, coup_e)
    detaphi = compute_detaphi(n_c, eta)
    detaphi_prime = compute_detaphiprime(n_c, eta)
    omega = Float64((phi_prime - phi) / L^2)
    detaomega = Float64((detaphi_prime - detaphi) / L^2)

    # Initialize 
    r = zeros(Float64, 3)
    detar = fill(Float64(detaphi / L), 3)
    q0 = fill(Float64(omega * t), 3) #q_k
    q1 = zeros(Float64, 3) #\tilde{q}_k
    q2 = zeros(Float64, 3) #hat{q}_k
    b = zeros(Complex{Float64}, 3)
    c = zeros(Complex{Float64}, 3)
    detab = zeros(Complex{Float64}, 3)
    detac = zeros(Complex{Float64}, 3)
    p0k = Complex{Float64}(im * sin(omega))
    pp0k = Complex{Float64}(im * cos(omega)) #dp0k/dω
    temp = 0.5 * c_sw * detaomega * pp0k
    temp1 = 0.5 * c_sw * p0k

    gamma_sumb = @MMatrix zeros(Complex{Float64}, 4, 4)
    gamma_sumc = @MMatrix zeros(Complex{Float64}, 2, 4)#[3:4,1:4]
    gamma_sumdetab = @MMatrix zeros(Complex{Float64}, 4, 4)
    gamma_sumdetac = @MMatrix zeros(Complex{Float64}, 2, 4)#[3:4,1:4]
    # Compute coefficient function d and derivative
    sum_q2 = Float64(0)
    detad = Float64(0)

    # Compute coefficients and derivatives
    for k in 1:3
        begin
            r[k] = phi / L + p[k]
            q0[k] += r[k]
            q1[k] = sin(q0[k])
            q2[k] = 2 * sin(0.5 * q0[k])

            tempk1 = im * q1[k]
            b[k] = tempk1 - temp1
            c[k] = tempk1 + temp1

            tempk = im * (t * detaomega + detar[k]) * cos(q0[k])
            detab[k] = tempk - temp
            detac[k] = tempk + temp

            gamma_sumb += gamma[k] * b[k]
            gamma_sumc[1:2, 1:2] += gamma[k][3:4, 1:2] * c[k]
            gamma_sumdetab += gamma[k] * detab[k]
            gamma_sumdetac[1:2, 1:2] += gamma[k][3:4, 1:2] * detac[k]

            sum_q2 += q2[k]^2
            detad += (t * detaomega + detar[k]) * q1[k]
        end
    end

    d = 1 + m + 0.5 * sum_q2

    # Calculate B and B_prime using P_+ and P_- partitions
    B = @MMatrix zeros(Complex{Float64}, 4, 4)
    B_prime = @MMatrix zeros(Complex{Float64}, 4, 4)


    # Optimised gamma_sumc,gamma_sumdetac to 2x4 matrices
    B[3:4, 1:4] = gamma_sumc * (id - gamma_sumb) + d^2 * id[3:4, 1:4]
    B[1:2, 1:4] = id[1:2, 1:4] - gamma_sumb[1:2, 1:4]

    B_prime[3:4, 1:4] = gamma_sumdetac * (id - gamma_sumb) - gamma_sumc * gamma_sumdetab + 2 * d * detad * id[3:4, 1:4]
    B_prime[1:2, 1:4] = -gamma_sumdetab[1:2, 1:4]




    return SMatrix{4,4}(B), SMatrix{4,4}(B_prime)
end



function symfactor(a::Int64, b::Int64, c::Int64)::Int64
    if a == b == c
        return 1
    elseif a == b || a == c || b == c
        return 3
    else
        return 6
    end
end


function Sum_trace(L::Int64, coup_e)
    sums = Float64(0)
    #sum2 = Float64(0) #sum for derivative over mass
    L_s = Int64(L * rho)
    p = Array{Float64}(undef, 3)

    B = @SMatrix zeros(Complex{Float64}, 4, 4)
    B_prime = @SMatrix zeros(Complex{Float64}, 4, 4)

    M = @SMatrix zeros(Complex{Float64}, 4, 4)
    M_inv = @SMatrix zeros(Complex{Float64}, 4, 4)

    M_prime = @SMatrix zeros(Complex{Float64}, 4, 4)

    Mt = SMatrix{4,2}(B[1:4, 3:4])
    Mt_prime = SMatrix{4,2}(B_prime[1:4, 3:4])


    for n in pyramidindices(3, L_s)
        for n_c in 1:3
            @inbounds begin

                p .= ((2Float64(pi)) .* n .+ theta) ./ L_s


                #Mt(1)=B(1)P_-
                B, B_prime = Calculate_BandB_prime(L, 1, p, n_c, coup_e)
                Mt = SMatrix{4,2}(B[1:4, 3:4])
                Mt_prime = SMatrix{4,2}(B_prime[1:4, 3:4])

                #recursion
                for t in 2:(L-1)
                    @inbounds begin
                        B, B_prime = Calculate_BandB_prime(L, t, p, n_c, coup_e)

                        Mt_prime = B_prime * Mt + B * Mt_prime
                        Mt = B * Mt
                    end
                end
                #project to subspace
                M = Mt[3:4, 1:2]
                M_prime = Mt_prime[3:4, 1:2]

                #=
                println("M")
                print_mat(M,3)
                println(" ");
                println("M_prime")
                print_mat(M_prime,3)
                println(" ");
                =#
                M_inv = inv(M)
                sums += symfactor(n[1], n[2], n[3]) * tr(SMatrix{2,2}(M_inv) * SMatrix{2,2}(M_prime)).re
            end
        end
    end

    return sums
end





using Plots

@time begin
    results = Dict{Int,Vector{Tuple{Float64,Float64}}}()
    for i in 0:1
        p_data = Tuple{Float64,Float64}[]
        L = Lmin + i * 4
        k_normc = 12 * L^2 * (sin(Float64(pi) / (3 * L^2)) + sin(2Float64(pi) / (3 * L^2)))

        coup_es = 0
        sumtrace = Sum_trace(L, coup_es)
        p_11_zero = sumtrace / k_normc

        push!(p_data, (Float64(0), Float64(0)))
        println(' ')
        println("p_11(L=$L,e=0)=  ", p_11_zero)
        
        for j in 1:n
            @inbounds begin
                alpha0 = j* alpha_max / n       #from eletron mass m_e scale to m_e + 100 MeV
                coup_es = sqrt(4 * Float64(pi) * alpha0)
                sumtrace = Sum_trace(L, coup_es)
                p_11 = sumtrace / k_normc

                #push!(p_data, (Float64(alpha0), Float64(p_11-p_11_zero)/abs(p_11_zero)))
                push!(p_data, (Float64(alpha0), Float64(p_11-p_11_zero)))
                println(' ')
                println("p_11(L=$L,alpha=$alpha0)=  ", p_11)
            end
        end

        results[L] = p_data
        println("="^30)
    end
    all_x = [point[1] for data in values(results) for point in data]
    x_min = minimum(all_x)
    x_max = maximum(all_x)

    plt = plot(title="p_11 percentage vs alpha (Float64 Precision)",
        xlabel="Coupling Strength (alpha)",
        ylabel="Oneloop p_11 deviation",
        legend=:topleft,
        xlims=(x_min, x_max),
        dpi=300)

    for (L, data) in sort(collect(results), by=x -> x[1])
        x = [d[1] for d in data]
        y = [d[2] for d in data]
        plot!(plt, x, y, label="L=$L", lw=2, marker=(:circle, 6))
    end

    display(plt)

end
#@time Sum_trace(16)
#@time Sum_trace_parallel(16)


#@btime Sum_trace(16)
#@btime Sum_trace_parallel(16)
