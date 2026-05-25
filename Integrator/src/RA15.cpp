// Faithful C++17 translation of Everhart's RA15 integrator (originally in FORTRAN).
// Reference: E. Everhart, "An efficient integrator that uses Gauss-Radau spacings",
//            in Dynamics of Comets: Their Origin and Evolution, 1985.
//
// Index convention: all arrays are 0-based here.  Comments note the
// equivalent 1-based FORTRAN indices where the mapping is non-obvious.

#include "RA15.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Gauss-Radau quadrature points scaled to [0, 1].
// H[0..7] correspond to FORTRAN H(1..8).
static constexpr double H[8] = {
    0.0,
    0.05626256053692215,
    0.18024069173689236,
    0.35262471711316964,
    0.54715362633055538,
    0.73421017721541053,
    0.88532094683909577,
    0.97752061356128750
};

void ra15(double* x, double* v, double tf, double xl, int ll,
          int nv, int nclass, ForceFunc force, const double* masses)
{
    // ----- flags -----
    const bool ncl = (nclass == 1);   // y'  = F(y,t)
    const bool npq = (nclass < 2);    // conservative or first-order
    const bool nes = (ll < 0);        // constant step size mode
    const double dir = (tf < 0.0) ? -1.0 : 1.0;
    const double pw  = 1.0 / 9.0;
    const double sr  = 1.4;           // max sequence-size growth factor

    xl = dir * std::abs(xl);
    const double ss = nes ? 0.0 : std::pow(10.0, -(double)ll);

    // ----- W and U integration-weight vectors -----
    // FORTRAN: W(N-1) = 1/(N + N^2),  U(N-1) = 1/N   for N = 2..8
    double W[7], U[7];
    for (int n = 2; n <= 8; ++n) {
        W[n - 2] = 1.0 / (double)(ncl ? n : n + n * n);
        U[n - 2] = 1.0 / (double)n;
    }
    const double w1 = ncl ? 1.0 : 0.5;

    // ----- C, D, R coefficient arrays (21 elements each) -----
    // FORTRAN uses 1-based indexing; here we use 0-based (FORTRAN index i → C++ index i-1).
    // NW(k+1) in FORTRAN (1-based, values: 0,0,1,3,6,10,15,21) → NW[k] below (0-based, k=3..7).
    static constexpr int NW[8] = { 0, 0, 1, 3, 6, 10, 15, 21 };

    double C[21], D[21], R[21];
    C[0] = -H[1];
    D[0] =  H[1];
    R[0] = 1.0 / (H[2] - H[1]);

    int la = 1, lc = 1;   // 1-based positions (converted to 0-based on access)
    for (int k = 3; k <= 7; ++k) {
        const int lb = la;
        la = lc + 1;
        lc = NW[k];   // FORTRAN LC = NW(K+1)

        C[la - 1] = -H[k - 1] * C[lb - 1];
        C[lc - 1] =  C[la - 2] - H[k - 1];
        D[la - 1] =  H[1] * D[lb - 1];
        D[lc - 1] = -C[lc - 1];
        R[la - 1] = 1.0 / (H[k] - H[1]);
        R[lc - 1] = 1.0 / (H[k] - H[k - 1]);

        if (k == 3) continue;

        for (int l = 4; l <= k; ++l) {
            const int ld = la + l - 3;
            const int le = lb + l - 4;
            C[ld - 1] = C[le - 1] - H[k - 1] * C[le];
            D[ld - 1] = D[le - 1] + H[l - 2] * D[le];
            R[ld - 1] = 1.0 / (H[k] - H[l - 2]);
        }
    }

    // ----- working arrays: B, G, E, BD are 7 x nv -----
    // B(j,k) = b[j*nv + k],  j in [0,6],  k in [0,nv-1]
    std::vector<double> b(7 * nv, 0.0), g(7 * nv, 0.0);
    std::vector<double> e(7 * nv, 0.0), bd(7 * nv, 0.0);
    std::vector<double> f1(nv), fj(nv), y(nv), z(nv);

    auto B  = [&](int j, int k) -> double& { return b[j * nv + k]; };
    auto G  = [&](int j, int k) -> double& { return g[j * nv + k]; };
    auto E  = [&](int j, int k) -> double& { return e[j * nv + k]; };
    auto BD = [&](int j, int k) -> double& { return bd[j * nv + k]; };

    if (ncl)
        for (int k = 0; k < nv; ++k) v[k] = 0.0;

    // ----- initial sequence size -----
    double tp = 0.1 * dir;
    if (xl != 0.0) tp = xl;
    if (nes)       tp = xl;
    if (tp / tf > 0.5) tp = 0.5 * tf;

    int  ncount = 0;
    double hv   = 0.0;

    // =========================================================
    //  Outer restart loop  (FORTRAN label 4000)
    //  Runs at most ~10 times, only ever needed on the very
    //  first sequence if the chosen step size was too large.
    // =========================================================
    while (true)
    {
        int  ns   = 0;
        double tm = 0.0;
        bool nper = false;
        bool nsf  = false;
        int  ni   = 6;

        force(x, v, 0.0, f1.data(), masses);

        // =====================================================
        //  Inner sequence loop  (FORTRAN label 722)
        // =====================================================
        while (true)
        {
            // Predict G-values from B-values (Eq. 2.7)
            for (int k = 0; k < nv; ++k) {
                G(0,k) = B(0,k) + D[ 0]*B(1,k) + D[ 1]*B(2,k) + D[ 3]*B(3,k) + D[ 6]*B(4,k) + D[10]*B(5,k) + D[15]*B(6,k);
                G(1,k) =          B(1,k)        + D[ 2]*B(2,k) + D[ 4]*B(3,k) + D[ 7]*B(4,k) + D[11]*B(5,k) + D[16]*B(6,k);
                G(2,k) =          B(2,k)        + D[ 5]*B(3,k) + D[ 8]*B(4,k) + D[12]*B(5,k) + D[17]*B(6,k);
                G(3,k) =          B(3,k)        + D[ 9]*B(4,k) + D[13]*B(5,k) + D[18]*B(6,k);
                G(4,k) =          B(4,k)        + D[14]*B(5,k) + D[19]*B(6,k);
                G(5,k) =          B(5,k)        + D[20]*B(6,k);
                G(6,k) =          B(6,k);
            }

            const double t    = tp;
            const double t2   = ncl ? t : t * t;
            const double tval = std::abs(t);

            // M substep-iteration loop (6 on first sequence, 2 thereafter)
            for (int m = 0; m < ni; ++m)
            {
                // Substeps at H[1]..H[7] (FORTRAN J=2..8)
                for (int j = 1; j < 8; ++j)
                {
                    const double s = H[j];
                    const double q = ncl ? 1.0 : s;

                    // Predict positions Y (and velocities Z for class 2) — Eq. 2.9/2.10
                    for (int k = 0; k < nv; ++k) {
                        const double a = W[2]*B(2,k) + s*(W[3]*B(3,k) + s*(W[4]*B(4,k)
                                       + s*(W[5]*B(5,k) + s*W[6]*B(6,k))));
                        y[k] = x[k] + q*(t*v[k] + t2*s*(f1[k]*w1
                             + s*(W[0]*B(0,k) + s*(W[1]*B(1,k) + s*a))));

                        if (!npq) {
                            const double az = U[2]*B(2,k) + s*(U[3]*B(3,k) + s*(U[4]*B(4,k)
                                            + s*(U[5]*B(5,k) + s*U[6]*B(6,k))));
                            z[k] = v[k] + s*t*(f1[k] + s*(U[0]*B(0,k) + s*(U[1]*B(1,k) + s*az)));
                        }
                    }

                    force(y.data(), npq ? v : z.data(), tm + s * t, fj.data(), masses);

                    // Update G and B arrays — Eqs. 2.4 and 2.5
                    // FORTRAN J (2-based) = j+1 here (j is 0-based into H)
                    const int J = j + 1;  // matches FORTRAN switch target labels 102..108

                    for (int k = 0; k < nv; ++k)
                    {
                        const double old_gjd = G(j - 1, k);
                        const double gk      = (fj[k] - f1[k]) / s;

                        // Divided-difference update of G[j-1] (Eq. 2.4)
                        switch (J) {
                        case 2: G(0,k) = gk; break;
                        case 3: G(1,k) = (gk - G(0,k)) * R[0]; break;
                        case 4: G(2,k) = ((gk - G(0,k)) * R[1] - G(1,k)) * R[2]; break;
                        case 5: G(3,k) = (((gk - G(0,k)) * R[3] - G(1,k)) * R[4] - G(2,k)) * R[5]; break;
                        case 6: G(4,k) = ((((gk - G(0,k)) * R[6]  - G(1,k)) * R[7]  - G(2,k)) * R[8]  - G(3,k)) * R[9]; break;
                        case 7: G(5,k) = (((((gk - G(0,k)) * R[10] - G(1,k)) * R[11] - G(2,k)) * R[12] - G(3,k)) * R[13] - G(4,k)) * R[14]; break;
                        case 8: G(6,k) = ((((((gk - G(0,k)) * R[15] - G(1,k)) * R[16] - G(2,k)) * R[17] - G(3,k)) * R[18] - G(4,k)) * R[19] - G(5,k)) * R[20]; break;
                        }

                        const double dg = G(j - 1, k) - old_gjd;  // improvement on G(j-1)
                        B(j - 1, k) += dg;

                        // Propagate improvement to lower B-values (Eq. 2.5)
                        switch (J) {
                        case 2: break;
                        case 3: B(0,k) += C[ 0]*dg; break;
                        case 4: B(0,k) += C[ 1]*dg; B(1,k) += C[ 2]*dg; break;
                        case 5: B(0,k) += C[ 3]*dg; B(1,k) += C[ 4]*dg; B(2,k) += C[ 5]*dg; break;
                        case 6: B(0,k) += C[ 6]*dg; B(1,k) += C[ 7]*dg; B(2,k) += C[ 8]*dg; B(3,k) += C[ 9]*dg; break;
                        case 7: B(0,k) += C[10]*dg; B(1,k) += C[11]*dg; B(2,k) += C[12]*dg; B(3,k) += C[13]*dg; B(4,k) += C[14]*dg; break;
                        case 8: B(0,k) += C[15]*dg; B(1,k) += C[16]*dg; B(2,k) += C[17]*dg; B(3,k) += C[18]*dg; B(4,k) += C[19]*dg; B(5,k) += C[20]*dg; break;
                        }
                    }
                } // substep j

                // Compute error estimator HV on final iteration (only when adaptive)
                if (!nes && m == ni - 1) {
                    hv = 0.0;
                    for (int k = 0; k < nv; ++k)
                        hv = std::max(hv, std::abs(B(6, k)));
                    hv = hv * W[6] / std::pow(tval, 7.0);
                }
            } // iteration m

            // ----- First-sequence restart check -----
            if (!nsf) {
                if (!nes) tp = std::pow(ss / hv, pw) * dir;
                if (nes)  tp = xl;

                if (!nes && tp / t <= 1.0) {
                    // New step is smaller than what we just tried → restart
                    tp *= 0.8;
                    if (++ncount > 10) return;
                    break;  // exit sequence loop → re-enter restart loop
                }
                nsf = true;
            }

            // ----- Update X, V at end of sequence (Eqs. 2.11, 2.12) -----
            for (int k = 0; k < nv; ++k) {
                x[k] += v[k]*t + t2*(f1[k]*w1
                      + B(0,k)*W[0] + B(1,k)*W[1] + B(2,k)*W[2] + B(3,k)*W[3]
                      + B(4,k)*W[4] + B(5,k)*W[5] + B(6,k)*W[6]);
                if (!ncl)
                    v[k] += t*(f1[k]
                          + B(0,k)*U[0] + B(1,k)*U[1] + B(2,k)*U[2] + B(3,k)*U[3]
                          + B(4,k)*U[4] + B(5,k)*U[5] + B(6,k)*U[6]);
            }
            tm += t;
            ++ns;

            if (nper) return;  // integration complete

            // ----- Force at new position + new sequence size -----
            force(x, v, tm, f1.data(), masses);

            if (!nes) {
                tp = dir * std::pow(ss / hv, pw);
                if (tp / t > sr) tp = t * sr;
            } else {
                tp = xl;
            }

            // Clamp last sequence to exactly reach tf
            if (dir * (tm + tp) >= dir * tf - 1.0e-8) {
                tp   = tf - tm;
                nper = true;
            }

            // ----- Predict B-values for next sequence (Eq. 2.13) -----
            const double q  = tp / t;
            const double q2 = q * q,  q3 = q * q2, q4 = q * q3;
            const double q5 = q * q4, q6 = q * q5, q7 = q * q6;

            for (int k = 0; k < nv; ++k) {
                // BD = correction between prediction and reality (not on first sequence)
                if (ns > 1)
                    for (int j = 0; j < 7; ++j)
                        BD(j, k) = B(j, k) - E(j, k);

                E(0,k) = q *(B(0,k) +  2*B(1,k) +  3*B(2,k) +  4*B(3,k) +  5*B(4,k) +  6*B(5,k) +  7*B(6,k));
                E(1,k) = q2*(           B(1,k) +  3*B(2,k) +  6*B(3,k) + 10*B(4,k) + 15*B(5,k) + 21*B(6,k));
                E(2,k) = q3*(                      B(2,k) +  4*B(3,k) + 10*B(4,k) + 20*B(5,k) + 35*B(6,k));
                E(3,k) = q4*(                                 B(3,k) +  5*B(4,k) + 15*B(5,k) + 35*B(6,k));
                E(4,k) = q5*(                                            B(4,k) +  6*B(5,k) + 21*B(6,k));
                E(5,k) = q6*(                                                       B(5,k) +  7*B(6,k));
                E(6,k) = q7*B(6,k);

                for (int j = 0; j < 7; ++j)
                    B(j, k) = E(j, k) + BD(j, k);
            }

            ni = 2;  // two iterations per sequence from here on
        } // sequence loop
    } // restart loop
}
