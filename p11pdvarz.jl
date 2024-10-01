#using Pkg; Pkg.add("SparseArrays")
#using SparseArrays
using LinearAlgebra
using Printf
using BenchmarkTools
using DoubleFloats


#BLAS.set_num_threads(4)  # Set the number of threads 


using SymmetricTensors

# Define the parameters
const eta =   Double64(0)
const nu =   Double64(0)
Lmin = 27
Lmax = 32
const theta =   Double64(pi)/5
const c_sw =   Double64(1)
const rho = 1

println("eta=$eta   nu=$nu   Lmin=$Lmin   Lmax=$Lmax   m=exp(1/L)-1   theta=$theta   c_sw=$c_sw   L(space)= $rho *L(time) ")




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
const phi1 = eta - Double64(pi) /   Double64(3)
const phi2 = eta * (nu -   Double64(1) /   Double64(2))
const phi3 = -eta * (nu +   Double64(1) /   Double64(2)) + Double64(pi) /   Double64(3)

# Calculate the phi prime values
const phi1_prime = -phi1 -   Double64(4) * Double64(pi) /   Double64(3)
const phi2_prime = -phi3 +   Double64(2) * Double64(pi) /   Double64(3)
const phi3_prime = -phi2 +   Double64(2) * Double64(pi) /   Double64(3)

# Create arrays for phi and phi prime
const phis,phis_prime  = [phi1, phi2, phi3],[phi1_prime, phi2_prime, phi3_prime]


# Function to calculate eta derivatives
const detaphis = [  Double64(1), nu -   Double64(1) /   Double64(2), -(nu +   Double64(1) /   Double64(2))]
const detaphis_prime = [-  Double64(1), nu +   Double64(1) /   Double64(2), -(nu -   Double64(1) /   Double64(2))]




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
function Calculate_BandB_prime(L::Int64,t::Int64,p::Array{Double64},n_c::Int64)
    # Initialize color related parameters
    omega =  Double64((phis_prime[n_c]- phis[n_c])/L^2) 
    detaomega = Double64((detaphis_prime[n_c] - detaphis[n_c])/L^2)

    # Initialize 
    r = zeros(Double64,3)
    detar = fill(Double64(detaphis[n_c] / L) ,3)
    q0 = fill(Double64(omega * t),3) #q_k
    q1 = zeros(Double64,3) #\tilde{q}_k
    q2 = zeros(Double64,3) #hat{q}_k
    b = zeros(Complex{Double64}, 3)
    c = zeros(Complex{Double64}, 3)
    detab = zeros(Complex{Double64}, 3)
    detac = zeros(Complex{Double64}, 3)
    p0k = Complex{Double64}(im*sin(omega))
    pp0k = Complex{Double64}(im*cos(omega)) #dp0k/dω
    temp = 0.5 * c_sw * detaomega * pp0k  
    temp1 = 0.5 * c_sw * p0k  

    gamma_sumb = zeros(Complex{Double64},4,4)
    gamma_sumc = zeros(Complex{Double64},2,4)#[3:4,1:4]
    gamma_sumdetab = zeros(Complex{Double64},4,4)
    gamma_sumdetac = zeros(Complex{Double64},2,4)#[3:4,1:4]
    # Compute coefficient function d and derivative
    sum_q2 = Double64(0)
    detad = Double64(0)
    
    # Compute coefficients and derivatives
    for k in 1:3
          begin
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
            gamma_sumc[1:2,1:2] += gamma[k][3:4,1:2] * c[k]
            gamma_sumdetab += gamma[k] * detab[k]
            gamma_sumdetac[1:2,1:2] += gamma[k][3:4,1:2] * detac[k]

            sum_q2 += q2[k]^2
            detad += (t * detaomega + detar[k]) * q1[k]
        end
    end
    m = Double64(exp(1/L))-1
    #println("m=",m)
    
    d = 1 + m + 0.5*sum_q2  

    # Calculate B and B_prime using P_+ and P_- partitions
    B = zeros(Complex{Double64},4,4)
    B_prime = zeros(Complex{Double64},4,4)
    

    # Optimised gamma_sumc,gamma_sumdetac to 2x4 matrices
    B[3:4,1:4] =  gamma_sumc * (id - gamma_sumb) + d^2 * id[3:4,1:4]
    B[1:2,1:4] = id[1:2,1:4] - gamma_sumb[1:2,1:4]

    B_prime[3:4,1:4] = gamma_sumdetac * (id - gamma_sumb) - gamma_sumc * gamma_sumdetab + 2 * d * detad * id[3:4,1:4] 
    B_prime[1:2,1:4] = -gamma_sumdetab[1:2,1:4]

    B_dot  = P_minus*2*d
    B_dotprime = P_minus*2*detad

    return B, B_prime,B_dot,B_dotprime

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
    sum =   Double64(0) 
    sum2 = Double64(0) #sum for derivative over mass
    L_s = Int64(L * rho)
    p = Array{Double64}(undef, 3)
    for n in pyramidindices(3, L_s)
        for n_c in 1:3
            @inbounds begin
                
                p .= ( (2Double64(pi)) .* n .+ theta) ./ L_s


                #Mt(1)=B(1)P_-
                B, B_prime,B_dot, B_dotprime = Calculate_BandB_prime(L, 1, p, n_c)
                Mt = B[1:4,3:4]
                Mt_prime = B_prime[1:4,3:4]
                Mt_dot = B_dot[1:4,3:4]
                Mt_dotprime = B_dotprime[1:4,3:4]
                #recursion
                for t in 2:(L-1)
                    @inbounds begin
                        B, B_prime,B_dot, B_dotprime = Calculate_BandB_prime(L, t, p, n_c)
                        Mt_dotprime = B_dotprime * Mt + B_dot * Mt_prime + B_prime * Mt_dot + B * Mt_dotprime
                        Mt_dot = B_dot * Mt + B * Mt_dot
                        Mt_prime = B_prime * Mt + B * Mt_prime
                        Mt = B * Mt
                    end
                end
                #project to subspace
                M = Mt[3:4, 1:2]
                M_prime = Mt_prime[end-1:end, end-1:end]
                M_dot = Mt_dot[end-1:end, end-1:end]
                M_dotprime = Mt_dotprime[end-1:end, end-1:end]
                #=
                println("M")
                print_mat(M,3)
                println(" ");
                println("M_prime")
                print_mat(M_prime,3)
                println(" ");
                =#
                M_inv = inv(M)
                sum += symfactor(n[1], n[2], n[3]) * tr(M_inv * M_prime).re
                sum2 += symfactor(n[1], n[2], n[3]) * (tr(-M_inv * M_dot * M_inv * M_prime).re + tr(M_inv*M_dotprime).re)
            end
        end
    end

    return sum,sum2
end




p_11_array = Array{ Double64}(undef, Lmax - Lmin + 1)
p_11_dot_array = Array{ Double64}(undef, Lmax - Lmin + 1)


@time begin

for l in Lmin:Lmax
    @inbounds begin
        L = l
        k_normc = 12 * L^2 * (sin(Double64(pi) / (3 * L^2)) + sin( 2Double64(pi) / (3 * L^2)))
        sum,sumdot = Sum_trace(L)
        p_11_array[l-Lmin+1] =sum/k_normc
        p_11_dot_array[l-Lmin+1] =sumdot/k_normc
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
for l in 1:Lmax-Lmin+1
    @inbounds begin
        L = l-1+Lmin
        println(' ')
        println("dot(p)_11(L=$L)=  ",p_11_dot_array[l])
    end
end



end
#@time Sum_trace(16)
#@time Sum_trace_parallel(16)


#@btime Sum_trace(16)
#@btime Sum_trace_parallel(16)








