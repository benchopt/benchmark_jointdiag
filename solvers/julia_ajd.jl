using Diagonalizations, PosDefManifold

function solve_ajd(C::Array{Float64,3}, n_iter::Int)
    Cset = ℍVector([ℍ(C[s, :, :]) for s=1:size(C, 1)])
    out = ajd(Cset; algorithm=:LogLike, simple=true, maxiter=n_iter)
    return out.iF
end
