#include "Force.h"
#include <cmath>

// Pairwise gravitational accelerations for 5 coplanar bodies (G = 1).
// Uses all C(5,2) = 10 pairs; each pair's inverse-cube distance is computed
// once and reused for both members.
void computeForce(const double* X, const double* /*V*/, double /*t*/,
                  double* F, const double* M)
{
    static constexpr int NB = 5;

    double rh[NB][NB] = {};   // rh[i][j] = 1 / |r_i - r_j|^3

    for (int n = 0; n < NB; ++n) {
        for (int l = n + 1; l < NB; ++l) {
            const double dx = X[2*n]   - X[2*l];
            const double dy = X[2*n+1] - X[2*l+1];
            const double r2 = dx*dx + dy*dy;
            rh[n][l] = rh[l][n] = 1.0 / (r2 * std::sqrt(r2));
        }
    }

    for (int n = 0; n < NB; ++n) {
        F[2*n]   = 0.0;
        F[2*n+1] = 0.0;
        for (int l = 0; l < NB; ++l) {
            if (l == n) continue;
            F[2*n]   += M[l] * (X[2*l]   - X[2*n])   * rh[n][l];
            F[2*n+1] += M[l] * (X[2*l+1] - X[2*n+1]) * rh[n][l];
        }
    }
}
