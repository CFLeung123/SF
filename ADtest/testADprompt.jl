using ForwardDiff, StaticArrays, LinearAlgebra
const P_minus = SMatrix{4,4}(Diagonal([0, 0, 1, 1]))

function build_M(n_c::Int, L::Int, eta, m0)
    s = P_minus * (SMatrix{4,4}(1:16) * eta^2 * m0 + SMatrix{4,4}(1:16) * eta * m0^2) * P_minus
    s2 = SMatrix{2,2}(s[3:4, 3:4])
    inv(s2)
end
function dMdeta(η_val, m0_val)
    ForwardDiff.derivative(η -> build_M(1,2,η,m0_val),η_val)
end

function M(m::Real, η::Real)
    return [
        m^2 * η           im * m^2 * η ;
        m^2 * η                (1 + im) * m^2 * η 
    ]
end

# 
function mixed_derivative_stepwise(m0, η0)
    dM_dm(m, η) = ForwardDiff.derivative(m -> M(m, η), m)
    ForwardDiff.derivative(η -> dM_dm(m0, η), η0)
end



#
m_test, η_test = 1, 0
result_step = mixed_derivative_stepwise(m_test, η_test)  # ≈ 2 * 1.5 + 0.8*exp(1.5 * 0.8) + 1.5 * 0.8*exp(1.5 * 0.8)
dMdeta(1.0, 2.0), mixed_derivative_stepwise(m_test, η_test) 
