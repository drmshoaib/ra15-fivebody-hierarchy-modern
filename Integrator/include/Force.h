#pragma once

// Gravitational force function for 5 coplanar bodies.
// X[0..9] = positions (x0,y0, x1,y1, ..., x4,y4)
// V[0..9] = velocities (unused for conservative NCLASS=-2 problems)
// t       = current time (unused — autonomous system)
// F[0..9] = output accelerations
// M[0..4] = masses
void computeForce(const double* X, const double* V, double t,
                  double* F, const double* M);
