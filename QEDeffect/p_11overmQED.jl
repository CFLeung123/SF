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
Lmin = 4
Lmax = 48
m = 0
const theta = Double64(pi) / 5
const c_sw = 1
const rho = 1
const coup_e = sqrt(4 * Double64(pi) * Double64(0.0072973525693))
Q = Double64(+2/3)   #uptype Q=+2/3 downtype Q=-1/3
# photon = \varphi * Q * \frac{e}{a}
const phiQED = Double64(0.01)
const phipQED = Double64(-0.01)
#const phiQED = 0 # QED off
#const phiprimeQED = 0 # QED off

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
function compute_phi(n_c::Int, eta, Q)
    if n_c == 1
        phi = eta - Double64(pi) / 3 + phiQED * coup_e * Q
    elseif n_c == 2
        phi = eta * (nu - 0.5) + phiQED * coup_e * Q
    elseif n_c == 3
        phi = -eta * (nu + 0.5) + Double64(pi) / 3 + phiQED * coup_e * Q
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phi
end

function compute_phiprime(n_c::Int, eta, Q)
    if n_c == 1
        phip = -(eta - Double64(pi) / 3) - Double64(pi) * 4 / 3
    elseif n_c == 2
        phip = -(-eta * (nu + 0.5) + Double64(pi) / 3) + Double64(pi) * 2 / 3
    elseif n_c == 3
        phip = -(eta * (nu - 0.5)) + Double64(pi) * 2 / 3
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phip
end
function compute_detaphi(n_c::Int, eta, Q)
    if n_c == 1
        detaphi = Double64(1)
    elseif n_c == 2
        detaphi = nu - Double64(1) / Double64(2)
    elseif n_c == 3
        detaphi = -(nu + Double64(1) / Double64(2))
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return detaphi
end

function compute_detaphiprime(n_c::Int, eta, Q)
    if n_c == 1
        detaphip = -Double64(1)
    elseif n_c == 2
        detaphip = nu + Double64(1) / Double64(2)
    elseif n_c == 3
        detaphip = -(nu - Double64(1) / Double64(2))
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return detaphip
end

#=
# Function to calculate eta derivatives
const detaphis = [ Double64(1), nu - Double64(1) / Double64(2), -(nu + Double64(1) / Double64(2))]
const detaphis_prime = [-Double64(1), nu + Double64(1) / Double64(2), -(nu - Double64(1) / Double64(2))]
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
function Calculate_BandB_prime(L::Int64, t::Int64, p::Array{Double64}, n_c::Int64,Q)
    # Initialize color related parameters
    phi = compute_phi(n_c,eta,Q)
    phi_prime = compute_phiprime(n_c,eta,Q)
    detaphi = compute_detaphi(n_c,eta,Q)
    detaphi_prime = compute_detaphiprime(n_c,eta,Q)
    omega = Double64((phi_prime - phi) / L^2)
    detaomega = Double64((detaphi_prime - detaphi) / L^2)

    # Initialize 
    r = zeros(Double64, 3)
    detar = fill(Double64(detaphi / L), 3)
    q0 = fill(Double64(omega * t), 3) #q_k
    q1 = zeros(Double64, 3) #\tilde{q}_k
    q2 = zeros(Double64, 3) #hat{q}_k
    b = zeros(Complex{Double64}, 3)
    c = zeros(Complex{Double64}, 3)
    detab = zeros(Complex{Double64}, 3)
    detac = zeros(Complex{Double64}, 3)
    p0k = Complex{Double64}(im * sin(omega))
    pp0k = Complex{Double64}(im * cos(omega)) #dp0k/dω
    temp = 0.5 * c_sw * detaomega * pp0k
    temp1 = 0.5 * c_sw * p0k

    gamma_sumb = @MMatrix zeros(Complex{Double64}, 4, 4)
    gamma_sumc = @MMatrix zeros(Complex{Double64}, 2, 4)#[3:4,1:4]
    gamma_sumdetab = @MMatrix zeros(Complex{Double64}, 4, 4)
    gamma_sumdetac = @MMatrix zeros(Complex{Double64}, 2, 4)#[3:4,1:4]
    # Compute coefficient function d and derivative
    sum_q2 = Double64(0)
    detad = Double64(0)

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
    B = @MMatrix zeros(Complex{Double64}, 4, 4)
    B_prime = @MMatrix zeros(Complex{Double64}, 4, 4)
    B_dot = @MMatrix zeros(Complex{Double64}, 4, 4)
    B_dotprime = @MMatrix zeros(Complex{Double64}, 4, 4)

    # Optimised gamma_sumc,gamma_sumdetac to 2x4 matrices
    B[3:4, 1:4] = gamma_sumc * (id - gamma_sumb) + d^2 * id[3:4, 1:4]
    B[1:2, 1:4] = id[1:2, 1:4] - gamma_sumb[1:2, 1:4]

    B_prime[3:4, 1:4] = gamma_sumdetac * (id - gamma_sumb) - gamma_sumc * gamma_sumdetab + 2 * d * detad * id[3:4, 1:4]
    B_prime[1:2, 1:4] = -gamma_sumdetab[1:2, 1:4]

    B_dot = P_minus * 2 * d
    B_dotprime = P_minus * 2 * detad

    return SMatrix{4,4}(B), SMatrix{4,4}(B_prime), SMatrix{4,4}(B_dot), SMatrix{4,4}(B_dotprime)

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


function Sum_trace(L::Int64,Q)
    sum = Double64(0)
    sum2 = Double64(0) #sum for derivative over mass
    L_s = Int64(L * rho)
    p = Array{Double64}(undef, 3)

    B = @SMatrix zeros(Complex{Double64}, 4, 4)
    B_dot = @SMatrix zeros(Complex{Double64}, 4, 4)
    B_prime = @SMatrix zeros(Complex{Double64}, 4, 4)
    B_dotprime = @SMatrix zeros(Complex{Double64}, 4, 4)
    M = @SMatrix zeros(Complex{Double64}, 4, 4)
    M_inv = @SMatrix zeros(Complex{Double64}, 4, 4)
    M_dot = @SMatrix zeros(Complex{Double64}, 4, 4)
    M_prime = @SMatrix zeros(Complex{Double64}, 4, 4)
    M_dotprime = @SMatrix zeros(Complex{Double64}, 4, 4)
    Mt = SMatrix{4,2}(B[1:4, 3:4])
    Mt_prime = SMatrix{4,2}(B_prime[1:4, 3:4])
    Mt_dot = SMatrix{4,2}(B_dot[1:4, 3:4])
    Mt_dotprime = SMatrix{4,2}(B_dotprime[1:4, 3:4])

    for n in pyramidindices(3, L_s)
        for n_c in 1:3
            @inbounds begin

                p .= ((2Double64(pi)) .* n .+ theta) ./ L_s


                #Mt(1)=B(1)P_-
                B, B_prime, B_dot, B_dotprime = Calculate_BandB_prime(L, 1, p, n_c ,Q)
                Mt = SMatrix{4,2}(B[1:4, 3:4])
                Mt_prime = SMatrix{4,2}(B_prime[1:4, 3:4])
                Mt_dot = SMatrix{4,2}(B_dot[1:4, 3:4])
                Mt_dotprime = SMatrix{4,2}(B_dotprime[1:4, 3:4])
                #recursion
                for t in 2:(L-1)
                    @inbounds begin
                        B, B_prime, B_dot, B_dotprime = Calculate_BandB_prime(L, t, p, n_c,Q)
                        Mt_dotprime = B_dotprime * Mt + B_dot * Mt_prime + B_prime * Mt_dot + B * Mt_dotprime
                        Mt_dot = B_dot * Mt + B * Mt_dot
                        Mt_prime = B_prime * Mt + B * Mt_prime
                        Mt = B * Mt
                    end
                end
                #project to subspace
                M = Mt[3:4, 1:2]
                M_prime = Mt_prime[3:4, 1:2]
                M_dot = Mt_dot[3:4, 1:2]
                M_dotprime = Mt_dotprime[3:4, 1:2]
                #=
                println("M")
                print_mat(M,3)
                println(" ");
                println("M_prime")
                print_mat(M_prime,3)
                println(" ");
                =#
                M_inv = inv(M)
                sum += symfactor(n[1], n[2], n[3]) * tr(SMatrix{2,2}(M_inv) * SMatrix{2,2}(M_prime)).re
                sum2 += symfactor(n[1], n[2], n[3]) * (tr(-SMatrix{2,2}(M_inv) * SMatrix{2,2}(M_dot) * SMatrix{2,2}(M_inv) * SMatrix{2,2}(M_prime)).re + tr(SMatrix{2,2}(M_inv) * SMatrix{2,2}(M_dotprime)).re)
            end
        end
    end

    return sum, sum2
end




p_11_array = Array{Double64}(undef, Lmax - Lmin + 1)
p_11_dot_array = Array{Double64}(undef, Lmax - Lmin + 1)


@time begin

    for l in Lmin:Lmax
        @inbounds begin
            L = l
            k_normc = 12 * L^2 * (sin(Double64(pi) / (3 * L^2)) + sin(2Double64(pi) / (3 * L^2)))
            sum, sumdot = Sum_trace(L,Q)
            p_11_array[l-Lmin+1] = sum / k_normc
            p_11_dot_array[l-Lmin+1] = sumdot / k_normc
            println("L=$L completed")
        end
    end
    for l in 1:Lmax-Lmin+1
        @inbounds begin
            L = l - 1 + Lmin
            println(' ')
            println("p_11(L=$L)=  ", p_11_array[l])
        end
    end
    for l in 1:Lmax-Lmin+1
        @inbounds begin
            L = l - 1 + Lmin
            println(' ')
            println("dot(p)_11(L=$L)=  ", p_11_dot_array[l])
        end
    end



end
#@time Sum_trace(16)
#@time Sum_trace_parallel(16)


#@btime Sum_trace(16)
#@btime Sum_trace_parallel(16)
