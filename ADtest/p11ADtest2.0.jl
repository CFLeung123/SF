#using Pkg; Pkg.add("SparseArrays")
#using SparseArrays
using LinearAlgebra
using Printf
using BenchmarkTools
using DoubleFloats
using StaticArrays

BLAS.set_num_threads(4)  # Set the number of threads 


using SymmetricTensors
using ForwardDiff

# Define the parameters
const nu = Double64(0)
Lmin = 4
Lmax = 16
const theta = Double64(pi) / 5
const c_sw = 1
const rho = 1
eta = Double64(0)
nu = Double64(0)
m0 = Double64(0)

println("Lmin=$Lmin   Lmax=$Lmax   theta=$theta   c_sw=$c_sw   L(space)= $rho *L(time) ")




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


function compute_phi(n_c::Int, eta, nu)
    if n_c == 1
        phi = eta -  Double64(pi)/3
    elseif n_c == 2
        phi = eta * (nu - 0.5)
    elseif n_c == 3
        phi = -eta * (nu + 0.5) + Double64(pi)/3
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phi
end

function compute_phiprime(n_c::Int, eta, nu)
    if n_c == 1
        phip = -(eta - Double64(pi)/3) - Double64(pi)*4 / 3
    elseif n_c == 2
        phip = -(-eta * (nu + 0.5) + Double64(pi)/3) + Double64(pi)*2 / 3
    elseif n_c == 3
        phip = -(eta * (nu - 0.5))   + Double64(pi)*2 / 3
    else
        error("Invalid n_c: must be 1, 2, or 3.")
    end
    return phip
end


function build_Bt(n_c::Int, t::Int, L::Int, eta, nu::Double64, m0, p::SVector{3,Double64})
    phi = compute_phi(n_c,eta,nu)
    phi_prime = compute_phiprime(n_c,eta,nu)

    # Initialize color related parameters
    omega = (phi_prime - phi) / L^2

    # Initialize 
    r =   zeros(Real, 3)
    q0 =   fill(omega * t, 3) #q_k
    q1 =   zeros(Real, 3) #\tilde{q}_k
    q2 =   zeros(Real, 3) #hat{q}_k
    b =   zeros(Complex{Real}, 3)
    c =   zeros(Complex{Real}, 3)
    p0k = im * sin(omega)
    temp1 = 0.5 * c_sw * p0k

    gamma_sumb =   zeros(Complex{Double64}, 4, 4)
    gamma_sumc =   zeros(Complex{Double64}, 4, 4)
    # Compute coefficient function d and derivative
    sum_q2 = Double64(0)

    # Compute coefficients and derivatives
    for k in 1:3
        @inbounds begin
            r[k] = phi / L + p[k]
            q0[k] += r[k]
            q1[k] = sin(q0[k])
            q2[k] = 2 * sin(0.5 * q0[k])

            tempk1 = im * q1[k]
            b[k] = tempk1 - temp1
            c[k] = tempk1 + temp1


            gamma_sumb += gamma[k] * b[k]
            gamma_sumc += gamma[k] * c[k]


            sum_q2 += q2[k]^2

        end
    end

    d = 1 + m0 + 0.5 * sum_q2


    B = P_minus * (gamma_sumc * (id - gamma_sumb) + d^2 * id) + P_plus * (id - gamma_sumb)

    return B

end


function build_M(n_c::Int, L::Int, eta, nu::Double64, m0, p::SVector{3,Double64})
    M4 = P_minus
    for t in 1:(L-1)
        Bt = build_Bt(n_c, t, L, eta, nu, m0, p)
        M4 = Bt * M4  # 注意矩阵乘法顺序
    end
    M4 = P_minus * M4
    M2 = SMatrix{2,2}(M4[3:4, 3:4])
    return M2
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

function build_Mprime(n_c::Int, L::Int, eta, nu::Double64, m0, p::SVector{3,Double64})
    ForwardDiff.derivative(eta_var -> build_M(n_c,L,eta_var,nu,m0,p), eta)
end

function build_Mdot(n_c::Int, L::Int, eta, nu::Double64, m0, p::SVector{3,Double64})
    ForwardDiff.derivative(m_var -> build_M(n_c,L,eta,nu,m_var,p), m0)
end

function build_Mdotprime(n_c::Int, L::Int, eta, nu::Double64, m0, p::SVector{3,Double64})
    dM_dm(m_var, eta_var) = ForwardDiff.derivative(m_var -> build_M(n_c,L,eta_var,nu,m_var,p), m_var)
    ForwardDiff.derivative(eta_var -> dM_dm(m0, eta_var), eta)
end

function det_2dfast(M::SMatrix{2,2,T}) where T
    return M[1,1]*M[2,2] - M[1,2]*M[2,1]
end

function Sum_det(L::Int64, eta, nu::Double64, m0)
    sums = Double64(0)
    sums2 = Double64(0) #sum for derivative over mass
    L_s = Int64(L * rho)
   
    for n in pyramidindices(3, L_s)
        for n_c in 1:3
            @inbounds begin

                p = SVector{3,Double64}(((2Double64(pi)) .* n .+ theta) ./ L_s)
               

                #compute matrices and derivatives
                M_static = SMatrix(build_M(n_c,L,eta,nu,m0,p))
                sums += symfactor(n[1], n[2], n[3]) * log(real(det_2dfast(M_static)))
            end
        end
    end

    return sums
end

function result(L,eta,nu,m0)
    d1(m0_var, eta_var) = ForwardDiff.derivative(eta_var -> Sum_det(L,eta_var,nu,m0_var), eta) # d_eta
    d2 = ForwardDiff.derivative(m0_var -> d1(m0_var, eta), m0) # d_m0 d_eta
    return d1(m0,eta),d2
end

p_11_array = Array{Double64}(undef, Lmax - Lmin + 1)
p_11_dot_array = Array{Double64}(undef, Lmax - Lmin + 1)


@time begin

    for l in Lmin:Lmax
        @inbounds begin
            L = l
            k_normc = 12 * L^2 * (sin(Double64(pi) / (3 * L^2)) + sin(2Double64(pi) / (3 * L^2)))
            deta_detsum, sumdot = result(L,eta,nu,m0)
            p_11_array[l-Lmin+1] = deta_detsum / k_normc
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
#=     for l in 1:Lmax-Lmin+1
        @inbounds begin
            L = l - 1 + Lmin
            println(' ')
            println("dot(p)_11(L=$L)=  ", p_11_dot_array[l])
        end
    end =#



end

