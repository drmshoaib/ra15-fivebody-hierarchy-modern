#pragma once

// Signature for any force/acceleration function usable with ra15.
// X[0..nv-1]  — positions
// V[0..nv-1]  — velocities (only needed for NCLASS=2; pass nullptr otherwise)
// t           — current time
// F[0..nv-1]  — output: accelerations
// masses      — body masses (passed through unchanged to the force function)
using ForceFunc = void(*)(const double*, const double*, double,
                          double*, const double*);

// Everhart RA15 integrator (15th-order Gauss-Radau).
//
//   x, v    — state vector on entry; updated in place on exit
//   tf      — integration span (may be negative for backward integration)
//   xl      — initial sequence size hint (0 = let RA15 choose)
//   ll      — accuracy control: ss = 10^(-ll) per sequence;
//              if ll < 0 a fixed step size |xl| is used instead
//   nv      — number of equations (positions only; velocities are derived)
//   nclass  — ODE class: -2 for y'' = F(y) [conservative, no velocity dependence]
//   force   — acceleration function
//   masses  — forwarded unchanged to force()
void ra15(double* x, double* v, double tf, double xl, int ll,
          int nv, int nclass, ForceFunc force, const double* masses);
