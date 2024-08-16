#using Pkg; Pkg.add("SparseArrays")
#using SparseArrays
using LinearAlgebra
using Printf
using BenchmarkTools
#using Quadmath
using DoubleFloats
#setprecision(BigFloat, 128)

#BLAS.set_num_threads(4)  # Set the number of threads 


using SymmetricTensors

# Define the parameters
const eta = Float64(0)
const nu = Float64(0)
Lmin = 4
Lmax = 16
const m = Float64(0.0)
const theta = Float64(pi)/Float64(5)
const c_sw = Float64(1)
const rho = 1

println("eta=$eta   nu=$nu   Lmin=$Lmin   Lmax=$Lmax   m=$m   theta=$theta   c_sw=$c_sw   L(space)= $rho *L(time) ")




# Define gamma matrices
    
const gamma0 = Diagonal([1, 1, -1, -1]) # gamma0
const gamma = [ [0 0 0 -im; 0 0 -im 0; 0 im 0 0; im 0 0 0],  # gamma1
[0 0 0 1; 0 0 -1 0; 0 -1 0 0; 1 0 0 0],  # gamma2
[0 0 im 0; 0 0 0 -im; -im 0 0 0; 0 im 0 0] # gamma3
]
const gamma5 = [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0] # gamma5

# Define 4x4 identity matrix 
const id =  Diagonal([1,1,1,1])
# Define P_+ and P_- constants
const P_plus = Int64.(0.5 * (id + gamma0))
const P_minus = Int64.(0.5 * (id - gamma0))



# Define background fields
# Calculate the phi  values
const phi1 = eta - Float64(pi) / Float64(3)
const phi2 = eta * (nu - Float64(1) / Float64(2))
const phi3 = -eta * (nu + Float64(1) / Float64(2)) + Float64(pi) / Float64(3)

# Calculate the phi prime values
const phi1_prime = -phi1 - Float64(4) * Float64(pi) / Float64(3)
const phi2_prime = -phi3 + Float64(2) * Float64(pi) / Float64(3)
const phi3_prime = -phi2 + Float64(2) * Float64(pi) / Float64(3)

# Create arrays for phi and phi prime
const phis,phis_prime  = [phi1, phi2, phi3],[phi1_prime, phi2_prime, phi3_prime]


# Function to calculate eta derivatives
const detaphis = [Float64(1), nu - Float64(1) / Float64(2), -(nu + Float64(1) / Float64(2))]
const detaphis_prime = [-Float64(1), nu + Float64(1) / Float64(2), -(nu - Float64(1) / Float64(2))]




#print complex matrix with n significant figures
function print_mat(matrix, n)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            real_part = round(real(matrix[i, j]), sigdigits=n)
            imag_part = round(imag(matrix[i, j]), sigdigits=n)
            if (imag_part != 0) && (real_part !=0)
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
function Calculate_BandB_prime(L::Int64,t::Int64,p::Array{Float64},n_c::Int64)
    # Initialize color related parameters
    omega =  Float64((phis_prime[n_c]- phis[n_c])/L^2) 
    detaomega = Float64((detaphis_prime[n_c] - detaphis[n_c])/L^2)

    # Initialize 
    r = zeros(Float64,3)
    detar = fill(Float64(detaphis[n_c] / L) ,3)
    q0 = fill(Float64(omega * t),3) #q_k
    q1 = zeros(Float64,3) #\tilde{q}_k
    q2 = zeros(Float64,3) #hat{q}_k
    b = zeros(Complex{Float64}, 3)
    c = zeros(Complex{Float64}, 3)
    detab = zeros(Complex{Float64}, 3)
    detac = zeros(Complex{Float64}, 3)
    p0k = Complex{Float64}(im*sin(omega))
    pp0k = Complex{Float64}(im*cos(omega)) #dp0k/dω
    temp = 0.5 * c_sw * detaomega * pp0k  
    temp1 = 0.5 * c_sw * p0k  

    gamma_sumb = zeros(Complex{Float64},4,4)
    gamma_sumc = zeros(Complex{Float64},4,4)
    gamma_sumdetab = zeros(Complex{Float64},4,4)
    gamma_sumdetac = zeros(Complex{Float64},4,4)
    # Compute coefficient function d and derivative
    sum_q2 = Float64(0)
    detad = Float64(0)
    
    # Compute coefficients and derivatives
    for k in 1:3
        @inbounds begin
            r[k] = phis[n_c] / L + p[k]   
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
            gamma_sumc += gamma[k] * c[k]
            gamma_sumdetab += gamma[k] * detab[k]
            gamma_sumdetac += gamma[k] * detac[k]

            sum_q2 += q2[k]^2
            detad += (t * detaomega + detar[k]) * q1[k]
        end
    end

    d = 1 + m + 0.5*sum_q2  


    B = P_minus * ( gamma_sumc * (id - gamma_sumb) + d^2 * id) + P_plus * (id - gamma_sumb)  

    B_prime = P_minus * ( gamma_sumdetac * (id - gamma_sumb) - (gamma_sumc) * (gamma_sumdetab) + 2 * d * detad * id) - P_plus * gamma_sumdetab  

    return B, B_prime

end



function symfactor(a::Int64,b::Int64,c::Int64)::Int64
    if a == b == c
        return 1
    elseif a == b || a == c || b == c
        return 3
    else
        return 6
    end
end


function Sum_trace(L::Int64)
    sum = Float64(0) 
    L_s = Int64(L * rho)
    p = Array{Float64}(undef, 3)
    for n in pyramidindices(3, L_s)
        for n_c in 1:3
            @inbounds begin
                
                p .= ( (2Float64(pi)) .* n .+ theta) ./ L_s


                #Mt(1)=B(1)
                Mt, Mt_prime = Calculate_BandB_prime(L, 1, p, n_c)
                #recursion
                for t in 2:(L-1)
                    @inbounds begin
                        B, B_prime = Calculate_BandB_prime(L, t, p, n_c)
                        Mt_prime, Mt = B_prime * Mt + B * Mt_prime, B * Mt
                    end
                end
                #project to subspace
                M4 = P_minus * Mt * P_minus
                M_prime4 = P_minus * Mt_prime * P_minus
                M, M_prime = M4[end-1:end, end-1:end], M_prime4[end-1:end, end-1:end]

                #=
                println("M")
                print_mat(M,3)
                println(" ");
                println("M_prime")
                print_mat(M_prime,3)
                println(" ");
                =#
                sum += symfactor(n[1], n[2], n[3]) * tr(inv(M) * M_prime).re

            end
        end
    end

    return sum
end



@time begin

p_11_array = Array{Float64}(undef, Lmax - Lmin + 1)

for l in Lmin:Lmax
    @inbounds begin
        L = l
        k_normc = 12 * L^2 * (sin(Float64(pi) / (3 * L^2)) + sin( 2Float64(pi) / (3 * L^2)))
        p_11_array[l-Lmin+1] = Sum_trace(L)/k_normc
        println("L=$L completed")
    end
end
for l in 1:Lmax-Lmin+1
    @inbounds begin
        L = l-1+Lmin
        println(' ')
        println("p_11(L=$L)=  ",p_11_array[l])
    end
end


end
#@time Sum_trace(16)
#@time Sum_trace_parallel(16)


#@btime Sum_trace(16)
#@btime Sum_trace_parallel(16)










