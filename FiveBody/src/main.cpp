#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "../../Integrator/include/Force.h"
#include "../../Integrator/include/RA15.h"

// ── Global constants ──────────────────────────────────────────────────────────
static constexpr int    NV         = 10;      // 5 bodies × 2 coordinates
static constexpr int    NB         = 5;       // number of bodies
static constexpr int    LL         = 8;       // RA15 accuracy: ss = 10^-8
static constexpr double TIMESTEP   = 0.001;   // time per RA15 call (integration sub-interval)
static constexpr double E0_FIXED   = -0.07;   // fixed total energy for all runs
static constexpr double ENERGY_TOL = 1.0e-5;  // relative energy-error tolerance

// ── Boundary parameters (scaled by 1/|E0| inside loadBoundaries) ─────────────
struct Boundaries {
    double b1, b2, b3, b4;
};

// ── Energy ───────────────────────────────────────────────────────────────────
static double calcPotential(const double* X, const double* M)
{
    double U = 0.0;
    for (int i = 0; i < NB; ++i)
        for (int j = i + 1; j < NB; ++j) {
            const double dx = X[2*i] - X[2*j];
            const double dy = X[2*i+1] - X[2*j+1];
            U += M[i] * M[j] / std::sqrt(dx*dx + dy*dy);
        }
    return U;
}

static double calcEnergy(const double* X, const double* V, const double* M)
{
    double K = 0.0;
    for (int i = 0; i < NB; ++i)
        K += 0.5 * M[i] * (V[2*i]*V[2*i] + V[2*i+1]*V[2*i+1]);
    return K - calcPotential(X, M);
}

// ── C₂ symmetry enforcement ("rest2" variant) ─────────────────────────────────
// Body 2 = −body 1,  body 3 = −body 0,  body 4 fixed at origin.
// This is a dynamical invariant when M[2]=M[1] and M[3]=M[0]; enforcing it
// each step prevents numerical drift from breaking the symmetry.
static void enforceSymmetry(double* X, double* V)
{
    X[4] = -X[2];  X[5] = -X[3];   // body 2 = −body 1
    X[6] = -X[0];  X[7] = -X[1];   // body 3 = −body 0
    X[8] =  0.0;   X[9] =  0.0;    // body 4 at origin

    V[4] = -V[2];  V[5] = -V[3];
    V[6] = -V[0];  V[7] = -V[1];
    V[8] =  0.0;   V[9] =  0.0;
}

// ── Hierarchy classification ──────────────────────────────────────────────────
// Returns 12, 13, 14, 23, or -1 (undetermined / transition zone).
//
//   r1  = |r₀| — distance of body 0 from origin
//   r2  = |r₁| — distance of body 1 from origin
//   r12 = |r₀ − r₁|
//   r13 = |r₀ − r₂| = |r₀ + r₁|  (because r₂ = −r₁ by C₂ symmetry)
static int identifyHierarchy(const double* X, const Boundaries& b)
{
    const double r1  = std::sqrt(X[0]*X[0] + X[1]*X[1]);
    const double r2  = std::sqrt(X[2]*X[2] + X[3]*X[3]);
    const double r12 = std::sqrt((X[0]-X[2])*(X[0]-X[2]) + (X[1]-X[3])*(X[1]-X[3]));
    const double r13 = std::sqrt((X[0]+X[2])*(X[0]+X[2]) + (X[1]+X[3])*(X[1]+X[3]));

    if (r2 < b.b2 && r1 > b.b1)  return 23;   // body 1 near centre, body 0 far out
    if (r1 < b.b4 && r2 > b.b3)  return 14;   // body 0 near centre, body 1 far out

    const bool outer = (r1 > b.b1 && r2 > b.b2) || (r1 > b.b4 && r2 > b.b3);
    if (outer)
        return (r12 < r13) ? 12 : 13;

    return -1;  // undetermined
}

// ── Hierarchy-change counting ─────────────────────────────────────────────────
// hc[12] accumulates directed transitions; see README §9 for the index map.
static void countHierarchyChange(int* hc, int prev, int cur)
{
    if (prev == -1 || cur == -1 || prev == cur) return;
    switch (prev) {
    case 12:
        if (cur == 13) { ++hc[0]; return; }
        if (cur == 14) { ++hc[1]; return; }
        if (cur == 23) { ++hc[2]; return; }
        break;
    case 13:
        if (cur == 12) { ++hc[3]; return; }
        if (cur == 14) { ++hc[4]; return; }
        if (cur == 23) { ++hc[5]; return; }
        break;
    case 14:
        if (cur == 12) { ++hc[6]; return; }
        if (cur == 13) { ++hc[7]; return; }
        if (cur == 23) { ++hc[8]; return; }
        break;
    case 23:
        if (cur == 12) { ++hc[9];  return; }
        if (cur == 13) { ++hc[10]; return; }
        if (cur == 14) { ++hc[11]; return; }
        break;
    }
}

// ── Symmetry-breaking stability check ────────────────────────────────────────
// Always false in the C₂-restricted variant (symmetry is enforced every step).
// Retained for non-restricted variants.
static bool isUnstable(const double* X)
{
    return ((X[0]+X[6])*(X[0]+X[4]) + (X[1]+X[7])*(X[1]+X[7]) >= 0.1 &&
            (X[2]+X[4])*(X[2]+X[6]) + (X[3]+X[5])*(X[3]+X[5]) >= 0.1);
}

// ── Initial-condition construction ────────────────────────────────────────────
// Sets X, V, M for the C₂-symmetric configuration given parameters
//   A  — x-position of body 0    B  — x-position of body 1
//   m0 — mass of bodies 0 & 3    m1 — mass of bodies 1 & 2
//   m4 — mass of central body    C0 — angular-momentum parameter
//
// Returns false if the discriminant is negative (configuration inaccessible
// at the fixed energy E0_FIXED).
static bool setupIC(double A, double B, double m0, double m1, double m4, double C0,
                    double* X, double* V, double* M)
{
    M[0] = m0;  M[1] = m1;  M[2] = m1;  M[3] = m0;  M[4] = m4;

    // All bodies start on the x-axis (y = 0)
    X[0] =  A;  X[1] = 0.0;   // body 0
    X[2] =  B;  X[3] = 0.0;   // body 1
    X[4] = -B;  X[5] = 0.0;   // body 2
    X[6] = -A;  X[7] = 0.0;   // body 3
    X[8] =  0.0; X[9] = 0.0;  // body 4

    // All x-velocities zero; body 4 at rest
    for (int k = 0; k < NV; ++k) V[k] = 0.0;

    // Angular momentum magnitude from C0: c = sqrt(C0 / |E0|)
    const double c = (E0_FIXED != 0.0) ? std::sqrt(C0 / (-E0_FIXED)) : 0.15;

    // Potential at the starting configuration
    const double U = calcPotential(X, M);

    // Discriminant for the quadratic in vy₀
    const double disc = -c*c + 4.0 * (A*A*m0 + B*B*m1) * (E0_FIXED + U);
    if (disc < 0.0) return false;

    // Solve two-constraint system (angular momentum + energy) for vy₀, vy₁
    const double sqrtTerm = std::sqrt(m0 * m1 * disc);
    V[1] = (A*c*m0 - B*sqrtTerm) / (2.0*m0*m0*A*A + 2.0*m1*m0*B*B);  // vy₀
    V[3] = (B*c*m1 + A*sqrtTerm) / (2.0*m0*m1*A*A + 2.0*m1*m1*B*B);  // vy₁

    // Symmetry-partner velocities
    V[5] = -V[3];   // vy₂ = −vy₁
    V[7] = -V[1];   // vy₃ = −vy₀

    return true;
}

// ── Single-IC integration ─────────────────────────────────────────────────────
// Returns:
//   0   stable (completed all steps)
//   1   unstable (symmetry-breaking detected — only active in non-rest2 mode)
//  -1   energy conservation failure
static int runIntegration(double* X, double* V, const double* M,
                          int nSteps, const Boundaries& bnd,
                          int* hc, double& t12, double& t13,
                          double& t14, double& t23)
{
    const double E0 = calcEnergy(X, V, M);
    std::fill(hc, hc + 12, 0);
    t12 = t13 = t14 = t23 = 0.0;

    int prevHie = identifyHierarchy(X, bnd);

    for (int step = 0; step < nSteps; ++step) {
        ra15(X, V, TIMESTEP, 0.0, LL, NV, -2, computeForce, M);
        enforceSymmetry(X, V);

        // Energy conservation check
        const double E = calcEnergy(X, V, M);
        if (std::abs((E - E0) / E0) > ENERGY_TOL)
            return -1;

        // Stability check (always passes in rest2 mode)
        if (isUnstable(X))
            return 1;

        // Hierarchy tracking
        const int curHie = identifyHierarchy(X, bnd);
        switch (curHie) {
        case 12: t12 += TIMESTEP; break;
        case 13: t13 += TIMESTEP; break;
        case 14: t14 += TIMESTEP; break;
        case 23: t23 += TIMESTEP; break;
        }
        if (curHie != prevHie)
            countHierarchyChange(hc, prevHie, curHie);
        if (curHie != -1)
            prevHie = curHie;
    }
    return 0;  // stable
}

// ── Resume-point detection ────────────────────────────────────────────────────
// Scans existing output files to find where a previous run stopped, so the
// batch can be resumed without repeating completed initial conditions.
struct StartPoint { int fileIdx; int firstRow; };

static StartPoint findStart(const std::vector<std::string>& outFiles, int nPerFile)
{
    int lastExisting = -1;
    for (int i = 0; i < static_cast<int>(outFiles.size()); ++i) {
        std::ifstream f(outFiles[i]);
        if (f.good()) lastExisting = i;
    }

    if (lastExisting < 0) return {0, 0};

    // Count data rows already written in the last existing output file
    std::ifstream f(outFiles[lastExisting]);
    int count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) ++count;
    }

    if (count >= nPerFile) return {lastExisting + 1, 0};   // file complete
    return {lastExisting, count};                           // resume mid-file
}

// ── Process one (boundary, IC, output) file triplet ──────────────────────────
static void processFileTriplet(const std::string& bndFile,
                                const std::string& icFile,
                                const std::string& outFile,
                                int nIC, int nSteps,
                                int firstRow, bool append)
{
    // Load boundary parameters and scale by 1/|E0|
    std::ifstream bIn(bndFile);
    if (!bIn) {
        std::cerr << "Cannot open boundary file: " << bndFile << "\n";
        return;
    }
    Boundaries bnd{};
    bIn >> bnd.b1 >> bnd.b2 >> bnd.b3 >> bnd.b4;
    const double scale = 1.0 / (-E0_FIXED);
    bnd.b1 *= scale;  bnd.b2 *= scale;
    bnd.b3 *= scale;  bnd.b4 *= scale;

    // Open IC file
    std::ifstream icIn(icFile);
    if (!icIn) {
        std::cerr << "Cannot open IC file: " << icFile << "\n";
        return;
    }

    // Open output file (append if resuming, overwrite if fresh)
    auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
    std::ofstream out(outFile, mode);
    if (!out) {
        std::cerr << "Cannot open output file: " << outFile << "\n";
        return;
    }

    // Write column header on fresh run
    if (!append) {
        out << "# A B flag"
               " hc12-13 hc12-14 hc12-23"
               " hc13-12 hc13-14 hc13-23"
               " hc14-12 hc14-13 hc14-23"
               " hc23-12 hc23-13 hc23-14"
               " t12 t13 t14 t23"
               " xf0 yf0 xf1 yf1\n";
    }

    std::cout << "  " << icFile << " -> " << outFile
              << "  (" << nIC << " ICs, " << nSteps << " steps each)\n";

    // Helper: read next data line from IC file, skipping blank lines and
    // lines beginning with '#'.  Returns false when the file is exhausted.
    auto nextICRow = [&](double& a1, double& a2, double& a3, double& a4,
                         double& a5, double& a6, double& a7, double& a8) -> bool {
        std::string line;
        while (std::getline(icIn, line)) {
            const size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream ss(line);
            if (ss >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8)
                return true;
        }
        return false;
    };

    for (int i = 0; i < nIC; ++i) {
        double a1, a2, a3, a4, a5, a6, a7, a8;
        if (!nextICRow(a1, a2, a3, a4, a5, a6, a7, a8)) break;

        if (i < firstRow) continue;   // skip already-completed rows

        if (a1 <= 0.0) {
            out << a1 << " " << a4
                << " -9999 0 0 0 0 0 0 0 0 0 0 0 0"
                   " 0 0 0 0  0 0 0 0\n";
            out.flush();
            continue;
        }

        const double A = a1, B = a4;
        const double m0 = a5, m1 = a6, m4 = a7, C0 = a8;

        double X[NV], V[NV], M[NB];
        if (!setupIC(A, B, m0, m1, m4, C0, X, V, M)) {
            // Discriminant < 0: configuration inaccessible at E0
            out << A << " " << B
                << " -9999 0 0 0 0 0 0 0 0 0 0 0 0"
                   " 0 0 0 0  0 0 0 0\n";
            out.flush();
            continue;
        }

        int    hc[12];
        double t12, t13, t14, t23;
        const int result = runIntegration(X, V, M, nSteps, bnd,
                                          hc, t12, t13, t14, t23);

        // Map internal return code to output flag
        int flag;
        switch (result) {
        case  0: flag =    0; break;   // stable
        case  1: flag =    1; break;   // symmetry-breaking unstable
        case -1: flag = -999; break;   // energy conservation failed
        default: flag = -9999;         // other failure
        }

        out << std::fixed << std::setprecision(6)
            << A << " " << B << " " << flag;
        for (int j = 0; j < 12; ++j) out << " " << hc[j];
        out << std::scientific << std::setprecision(6)
            << " " << t12 << " " << t13 << " " << t14 << " " << t23
            << "  " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]
            << "\n";
        out.flush();

        if ((i + 1) % 100 == 0)
            std::cout << "    " << (i + 1) << " / " << nIC << " done\r" << std::flush;
    }
    std::cout << "\n";
}

// ── Entry point ───────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    const std::string cfgPath = (argc > 1) ? argv[1] : "data/filenames.txt";

    std::ifstream cfg(cfgPath);
    if (!cfg) {
        std::cerr << "Cannot open configuration file: " << cfgPath << "\n"
                  << "Usage: FiveBody.exe [filenames.txt]\n";
        return 1;
    }

    int nFiles, nIC, nSteps;
    if (!(cfg >> nFiles >> nIC >> nSteps) || nFiles <= 0 || nIC <= 0 || nSteps <= 0) {
        std::cerr << "Invalid filenames.txt format.\n"
                  << "Expected: <numFiles> <ICs_per_file> <steps_per_IC>\n";
        return 1;
    }

    std::vector<std::string> bndFiles(nFiles), icFiles(nFiles), outFiles(nFiles);
    for (int i = 0; i < nFiles; ++i)
        cfg >> bndFiles[i] >> icFiles[i] >> outFiles[i];
    cfg.close();

    // Find where a previous run stopped
    const StartPoint sp = findStart(outFiles, nIC);

    std::cout << "Five-body hierarchical stability analyser\n"
              << "  Config:  " << cfgPath << "\n"
              << "  Files:   " << nFiles  << "\n"
              << "  ICs/file:" << nIC     << "  Steps/IC: " << nSteps << "\n"
              << "  Total integration time per IC: "
              << static_cast<double>(nSteps) * TIMESTEP << "\n";
    if (sp.fileIdx > 0 || sp.firstRow > 0)
        std::cout << "  Resuming from file " << sp.fileIdx
                  << ", row " << sp.firstRow << "\n";
    std::cout << "\n";

    for (int i = sp.fileIdx; i < nFiles; ++i) {
        const int  firstRow = (i == sp.fileIdx) ? sp.firstRow : 0;
        const bool append   = (i == sp.fileIdx) && (firstRow > 0);

        std::cout << "File " << (i + 1) << " / " << nFiles << "\n";
        processFileTriplet(bndFiles[i], icFiles[i], outFiles[i],
                           nIC, nSteps, firstRow, append);
    }

    std::cout << "Done.\n";
    return 0;
}
