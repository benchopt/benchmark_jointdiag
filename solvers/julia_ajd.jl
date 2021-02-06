using Diagonalizations, PosDefManifold


function solve_ajd(Xset, n_iter)
    aXset = ajd(Xset; algorithm=:LogLike; maxiter=n_iter)
    return aXset
end
