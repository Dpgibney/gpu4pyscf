/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gint/gint.h"
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define NG_PER_BLOCK      128

#define VV10_JTILES     4
// VV10_TILE (128) points per tile; VV10_JTILES J tiles per block (part 26)
#define VV10_TILE       128
#define VV10_WARPS      (VV10_TILE / 32)

// The symmetric version without per-j reductions: in a warp each lane keeps its i and meets the 32 j's of a chunk in
// turn, j = (lane + s) % 32 at step s, reading their data from shared memory; the j side's three sums travel with their
// j from lane to lane (a shuffle per step), so after 32 steps each is back on its lane, complete for the warp's i's.
__global__
static void vv10_fock_eval_UWE_rot_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                          const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                          const double* __restrict__ omega, const double* __restrict__ kappa,
                                          const int ngrids, const int jtiles)
{
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const int I = blockIdx.x;
    const int J0 = I + blockIdx.y * jtiles;
    if (J0 >= ntiles) {
        return;
    }
    const int J1 = min(J0 + jtiles, ntiles);
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int i = I * VV10_TILE + t;
    double omega_i = 1., kappa_i = 1., rho_weight_i = 0.;
    double3 r_i = {0., 0., 0.};
    if (i < ngrids) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        rho_weight_i = rho_weight[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }
    double U_i = 0, W_i = 0, E_i = 0;

    __shared__ double s_x[VV10_TILE], s_y[VV10_TILE], s_z[VV10_TILE];
    __shared__ double s_o[VV10_TILE], s_k[VV10_TILE], s_w[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
    const int next = (lane + 1) % 32;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_x[t] = grid_coord[jt * 3 + 0];
            s_y[t] = grid_coord[jt * 3 + 1];
            s_z[t] = grid_coord[jt * 3 + 2];
            s_o[t] = omega[jt];
            s_k[t] = kappa[jt];
            s_w[t] = rho_weight[jt];
        } else {
            s_x[t] = 0.; s_y[t] = 0.; s_z[t] = 0.;
            s_o[t] = 1.; s_k[t] = 1.; s_w[t] = 0.;
        }
        __syncthreads();
        const bool diag = J == I;
        for (int c = 0; c < VV10_WARPS; c++) {
            double U_j = 0, W_j = 0, E_j = 0;
            // in a diagonal tile the chunks below this warp's are the other warps' (j > i there)
            if (!diag || c >= warp) {
                for (int s = 0; s < 32; s++) {
                    const int jj = c * 32 + (lane + s) % 32;
                    const double dx = r_i.x - s_x[jj];
                    const double dy = r_i.y - s_y[jj];
                    const double dz = r_i.z - s_z[jj];
                    const double omega_j = s_o[jj];
                    const double kappa_j = s_k[jj];
                    const double rho_weight_j = s_w[jj];
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i * r_ij2 + kappa_i;
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_sum = g_ij + g_ji;
                    const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
                    const double Phi_ij = -g_ij_ji_sum_1;
                    // the i side: j >= i in a diagonal tile (j == i once); the j side: j > i there
                    if (!diag || jj >= t) {
                        const double E_ij = rho_weight_j * Phi_ij;
                        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
                        U_i += U_ij;
                        W_i += U_ij * r_ij2;
                        E_i += E_ij;
                    }
                    if (!diag || jj > t) {
                        const double E_ji = rho_weight_i * Phi_ij;
                        const double U_ji = E_ji * (g_sum + g_ji) * g_ij * g_ij_ji_sum_1;
                        U_j += U_ji;
                        W_j += U_ji * r_ij2;
                        E_j += E_ji;
                    }
                    // the sums of the j this lane meets next, from the lane that has it now
                    U_j = __shfl_sync(0xffffffff, U_j, next);
                    W_j = __shfl_sync(0xffffffff, W_j, next);
                    E_j = __shfl_sync(0xffffffff, E_j, next);
                }
            }
            // after 32 steps the sums of j = c*32 + lane are back on this lane
            s_acc[warp][0][c * 32 + lane] = U_j;
            s_acc[warp][1][c * 32 + lane] = W_j;
            s_acc[warp][2][c * 32 + lane] = E_j;
        }
        __syncthreads();
        if (jt < ngrids) {
            double u = 0, w = 0, e = 0;
            for (int k = 0; k < VV10_WARPS; k++) {
                u += s_acc[k][0][t];
                w += s_acc[k][1][t];
                e += s_acc[k][2][t];
            }
            atomicAdd(U + jt, u);
            atomicAdd(W + jt, w);
            atomicAdd(E + jt, e);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(U + i, U_i);
        atomicAdd(W + i, W_i);
        atomicAdd(E + i, E_i);
    }
}

__global__
static void vv10_scale_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E, const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngrids) {
        U[i] *= -1.5;
        W[i] *= -1.5;
        E[i] *= 1.5;
    }
}

__global__
static void vv10_fock_eval_omega_derivative_kernel(double* __restrict__ omega, double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega1 = sqrt(omega2);
    const double omega_1 = 1 / omega1;

    omega[i] = omega1;
    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;
}

__global__
static void vv10_grad_eval_E_grid_response_offdiagonal_kernel(double* __restrict__ Egr,
                                                              const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                              const double* __restrict__ omega, const double* __restrict__ kappa,
                                                              const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                              const int natoms, const int i_grid_begin, const int ngrids)
{
    const int i_unoffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i_unoffset >= ngrids || B_atom >= natoms)
        return;
    const int i = i_unoffset + i_grid_begin;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }
    if (i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ij = { r_i.x - r_j.x, r_i.y - r_j.y, r_i.z - r_j.z };
        const double r_ij2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
        const double Egr_ij = E_ij * dPhi_drj_over_Phi;

        Egr_i.x += Egr_ij * r_ij.x;
        Egr_i.y += Egr_ij * r_ij.y;
        Egr_i.z += Egr_ij * r_ij.z;
    }

    Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 1.5 * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 1.5 * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 1.5 * Egr_i.z;
}

__global__
static void vv10_hess_eval_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                         const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                         const double* __restrict__ omega, const double* __restrict__ kappa,
                                         const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double U_i = 0;
    double W_i = 0;
    double A_i = 0;
    double B_i = 0;
    double C_i = 0;
    double E_i = 0;

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_sum_1 = g_ji * g_ij_ji_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
        const double U_ij = E_ij * (g_ij + g_sum) * g_ij_sum_1;
        const double W_ij = U_ij * r_ij2;
        const double A_ij = E_ij * (g_ij * g_ij + g_ij * g_sum + g_sum * g_sum) * g_ij_sum_1 * g_ij_sum_1;
        const double B_ij = A_ij * r_ij2;
        const double C_ij = B_ij * r_ij2;

        U_i += U_ij;
        W_i += W_ij;
        A_i += A_ij;
        B_i += B_ij;
        C_i += C_ij;
        E_i += E_ij;
    }

    U[i] = -1.5 * U_i;
    W[i] = -1.5 * W_i;
    A[i] = (1.5 * 2) * A_i;
    B[i] = (1.5 * 2) * B_i;
    C[i] = (1.5 * 2) * C_i;
    E[i] = 1.5 * E_i;
}

__global__
static void vv10_hess_eval_omega_derivative_kernel(double* __restrict__ omega, double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   double* __restrict__ d2omega_drho2, double* __restrict__ d2omega_dgamma2, double* __restrict__ d2omega_drho_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_3 = rho_1 * rho_2;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega1 = sqrt(omega2);
    const double omega_1 = 1 / omega1;

    omega[i] = omega1;
    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;

    const double omega_3 = omega_1 / omega2;
    d2omega_drho2[i] = (-0.25 * four_pi_over_three * four_pi_over_three
                        + 12 * four_pi_over_three * C_factor * gamma2 * rho_5
                        + 6 * C_factor * C_factor * gamma2 * gamma2 * rho_5 * rho_5) * omega_3;
    d2omega_dgamma2[i] = four_pi_over_three * C_factor * rho_3 * omega_3;
    d2omega_drho_dgamma[i] = -C_factor * gamma_i * (4.5 * four_pi_over_three * rho_4
                                                    + 2 * C_factor * gamma2 * rho_4 * rho_5) * omega_3;
}

template <int n_trial_per_thread>
__global__
static void vv10_hess_eval_f_t_offdiagonal_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                                  const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                  const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                  const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                                  const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                                  const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_trial_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_trial_per_thread;
    if (i >= ngrids || i_trial_start >= ntrial)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    double f_rho_t_i[n_trial_per_thread] {0};
    double f_gamma_t_i[n_trial_per_thread] {0};
    // #pragma unroll
    // for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
    //     f_rho_t_i  [i_trial] = 0;
    //     f_gamma_t_i[i_trial] = 0;
    // }

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double domega_drho_j = domega_drho[j];
        const double domega_dgamma_j = domega_dgamma[j];
        const double dkappa_drho_j = dkappa_drho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
        const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
        const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_ji_1);

        const double f_rho_rho_ij = Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                              - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                              - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
        const double rho_domegadgamma_i_rij2 = rho_i * domega_dgamma_i * r_ij2;
        const double f_gamma_rho_ij = rho_domegadgamma_i_rij2 * Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
        const double rho_domegadgamma_j_rij2 = rho_j * domega_dgamma_j * r_ij2;
        const double f_rho_gamma_ij = rho_domegadgamma_j_rij2 * Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
        const double f_gamma_gamma_ij = rho_domegadgamma_i_rij2 * rho_domegadgamma_j_rij2 * Phi_ij * d2Phi_dgij_dgji_over_Phi;

        const double weight_j = grid_weight[j];

        #pragma unroll
        for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
            if (i_trial + i_trial_start >= ntrial) continue;
            const double   rho_t_j =   rho_t[(i_trial + i_trial_start) * ngrids + j];
            const double gamma_t_j = gamma_t[(i_trial + i_trial_start) * ngrids + j];
            f_rho_t_i  [i_trial] += weight_j * (  f_rho_rho_ij * rho_t_j +   f_rho_gamma_ij * gamma_t_j);
            f_gamma_t_i[i_trial] += weight_j * (f_gamma_rho_ij * rho_t_j + f_gamma_gamma_ij * gamma_t_j);
        }
    }

    #pragma unroll
    for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
        if (i_trial + i_trial_start >= ntrial) continue;
        f_rho_t  [(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_rho_t_i  [i_trial];
        f_gamma_t[(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_gamma_t_i[i_trial];
    }
}

__global__
static void vv10_hess_eval_f_t_diagonal_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                               const double* __restrict__ rho,
                                               const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                                               const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                               const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                                               const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                               const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    const double U_i = U[i];
    const double W_i = W[i];
    const double A_i = A[i];
    const double B_i = B[i];
    const double C_i = C[i];
    const double d2omega_drho2_i = d2omega_drho2[i];
    const double d2omega_dgamma2_i = d2omega_dgamma2[i];
    const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
    const double d2kappa_drho2_i = d2kappa_drho2[i];

    const double f_rho_rho_ii = 2 * domega_drho_i * W_i + 2 * dkappa_drho_i * U_i
                                + rho_i * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i * dkappa_drho_i * A_i
                                           + domega_drho_i * domega_drho_i * C_i + 2 * domega_drho_i * dkappa_drho_i * B_i);
    const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i * (d2omega_drho_dgamma_i * W_i
                                                                   + domega_dgamma_i * (dkappa_drho_i * B_i + domega_drho_i * C_i));
    const double f_rho_gamma_ii = f_gamma_rho_ii;
    const double f_gamma_gamma_ii = rho_i * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

    for (int i_trial = 0; i_trial < ntrial; i_trial++) {
        const double rho_t_i   =   rho_t[i_trial * ngrids + i];
        const double gamma_t_i = gamma_t[i_trial * ngrids + i];

        f_rho_t  [i_trial * ngrids + i] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
        f_gamma_t[i_trial * ngrids + i] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);
    }
}

__global__
static void vv10_hess_eval_EUW_grid_response_offdiagonal_kernel(double* __restrict__ Egr, double* __restrict__ Ugr, double* __restrict__ Wgr,
                                                                const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                                const double* __restrict__ omega, const double* __restrict__ kappa,
                                                                const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                                const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0 || i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };
    double3 Ugr_i = { 0, 0, 0 };
    double3 Wgr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = omega_i * g_ij_1 + omega_j * g_ji_1 + (omega_i + omega_j) * g_sum_1;
        const double d2Phi_dgij_drj_over_Phi = omega_i * g_ij_1 * g_ij_1 + (omega_i + omega_j) * g_sum_1 * g_sum_1;
        const double dPhi_dgij_over_Phi = g_sum_1 + g_ij_1;

        const double Egr_ij = E_ij * dPhi_drj_over_Phi;
        const double Ugr_ij = E_ij * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi);
        const double Wgr_ij = E_ij * (r_ij2 * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi) - dPhi_dgij_over_Phi);

        Egr_i.x += Egr_ij * r_ji.x;
        Egr_i.y += Egr_ij * r_ji.y;
        Egr_i.z += Egr_ij * r_ji.z;
        Ugr_i.x += Ugr_ij * r_ji.x;
        Ugr_i.y += Ugr_ij * r_ji.y;
        Ugr_i.z += Ugr_ij * r_ji.z;
        Wgr_i.x += Wgr_ij * r_ji.x;
        Wgr_i.y += Wgr_ij * r_ji.y;
        Wgr_i.z += Wgr_ij * r_ji.z;
    }

    // Factor of \pm 2 from Egr,Ugr,Wgr definition, 1.5 from phi definiton
    Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = -(2 * 1.5) * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = -(2 * 1.5) * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = -(2 * 1.5) * Egr_i.z;
    Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] =  (2 * 1.5) * Ugr_i.x;
    Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] =  (2 * 1.5) * Ugr_i.y;
    Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] =  (2 * 1.5) * Ugr_i.z;
    Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] =  (2 * 1.5) * Wgr_i.x;
    Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] =  (2 * 1.5) * Wgr_i.y;
    Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] =  (2 * 1.5) * Wgr_i.z;
}

template <int n_derivative_per_thread>
__global__
static void vv10_hess_eval_EUW_with_weight1_kernel(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                   const int ngrids, const int nderivative)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_derivative_per_thread;
    if (i >= ngrids || i_derivative_start >= nderivative)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double Ew_i[n_derivative_per_thread] {0};
    double Uw_i[n_derivative_per_thread] {0};
    double Ww_i[n_derivative_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
        const double W_ij = U_ij * r_ij2;

        #pragma unroll
        for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
            if (i_derivative + i_derivative_start >= nderivative) continue;
            const double weight_j = grid_weight1[(i_derivative + i_derivative_start) * ngrids + j];
            Ew_i[i_derivative] += weight_j * E_ij;
            Uw_i[i_derivative] += weight_j * U_ij;
            Ww_i[i_derivative] += weight_j * W_ij;
        }
    }

    #pragma unroll
    for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
        if (i_derivative + i_derivative_start >= nderivative) continue;
        Ew[(i_derivative + i_derivative_start) * ngrids + i] =  1.5 * Ew_i[i_derivative];
        Uw[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Uw_i[i_derivative];
        Ww[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Ww_i[i_derivative];
    }
}

__global__
static void vv10_hess_eval_D_B_in_double_grid_response_offdiagonal_kernel(double* __restrict__ D_B,
                                                                          const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                                          const double* __restrict__ omega, const double* __restrict__ kappa,
                                                                          const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                                          const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0 || i_associated_atom == B_atom) {
        D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double D_B_i[9] { 0,0,0, 0,0,0, 0,0,0, };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];


        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double omega_i_over_g_ij = omega_i * g_ij_1;
        const double omega_j_over_g_ji = omega_j * g_ji_1;
        const double omega_sum_over_g_sum = (omega_i + omega_j) * g_sum_1;
        const double omega_over_g_three_term_sum = omega_i_over_g_ij + omega_j_over_g_ji + omega_sum_over_g_sum;

        const double E_ij = rho_weight_j * Phi_ij;
        const double outer_product_prefactor = 2 * E_ij * (
            omega_over_g_three_term_sum * omega_over_g_three_term_sum
            + omega_i_over_g_ij * omega_i_over_g_ij
            + omega_j_over_g_ji * omega_j_over_g_ji
            + omega_sum_over_g_sum * omega_sum_over_g_sum
        );
        const double identity_prefactor = -E_ij * omega_over_g_three_term_sum;

        D_B_i[0] += outer_product_prefactor * r_ji.x * r_ji.x + identity_prefactor;
        D_B_i[1] += outer_product_prefactor * r_ji.x * r_ji.y;
        D_B_i[2] += outer_product_prefactor * r_ji.x * r_ji.z;
        D_B_i[3] += outer_product_prefactor * r_ji.y * r_ji.x;
        D_B_i[4] += outer_product_prefactor * r_ji.y * r_ji.y + identity_prefactor;
        D_B_i[5] += outer_product_prefactor * r_ji.y * r_ji.z;
        D_B_i[6] += outer_product_prefactor * r_ji.z * r_ji.x;
        D_B_i[7] += outer_product_prefactor * r_ji.z * r_ji.y;
        D_B_i[8] += outer_product_prefactor * r_ji.z * r_ji.z + identity_prefactor;
    }

    // Factor of 2 from D_B definition, 1.5 from phi definiton
    D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = -(2 * 1.5) * D_B_i[0];
    D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = -(2 * 1.5) * D_B_i[1];
    D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = -(2 * 1.5) * D_B_i[2];
    D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = -(2 * 1.5) * D_B_i[3];
    D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = -(2 * 1.5) * D_B_i[4];
    D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = -(2 * 1.5) * D_B_i[5];
    D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = -(2 * 1.5) * D_B_i[6];
    D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = -(2 * 1.5) * D_B_i[7];
    D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = -(2 * 1.5) * D_B_i[8];
}

// part 28: T_j = sum over i on other atoms of K_ij (r_i - r_j), each unordered pair once (zeroed before)
__global__
static void vv10_grad_grid_response_T_kernel(double* __restrict__ T, const double* __restrict__ grid_coord,
                                             const double* __restrict__ rho_weight, const double* __restrict__ omega,
                                             const double* __restrict__ kappa, const int* __restrict__ atom,
                                             const int ngrids, const int jtiles)
{
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const int I = blockIdx.x;
    const int J0 = I + blockIdx.y * jtiles;
    if (J0 >= ntiles) {
        return;
    }
    const int J1 = min(J0 + jtiles, ntiles);
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int i = I * VV10_TILE + t;
    double omega_i = 1., kappa_i = 1., rho_weight_i = 0.;
    double3 r_i = {0., 0., 0.};
    int atom_i = -1;
    if (i < ngrids) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        rho_weight_i = rho_weight[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
        atom_i = atom[i];
    }
    double Tix = 0, Tiy = 0, Tiz = 0;

    __shared__ double s_x[VV10_TILE], s_y[VV10_TILE], s_z[VV10_TILE];
    __shared__ double s_o[VV10_TILE], s_k[VV10_TILE], s_w[VV10_TILE];
    __shared__ int s_a[VV10_TILE];
    __shared__ double s_acc[VV10_WARPS][3][VV10_TILE];
    const int next = (lane + 1) % 32;

    for (int J = J0; J < J1; J++) {
        const int jt = J * VV10_TILE + t;
        if (jt < ngrids) {
            s_x[t] = grid_coord[jt * 3 + 0];
            s_y[t] = grid_coord[jt * 3 + 1];
            s_z[t] = grid_coord[jt * 3 + 2];
            s_o[t] = omega[jt];
            s_k[t] = kappa[jt];
            s_w[t] = rho_weight[jt];
            s_a[t] = atom[jt];
        } else {
            s_x[t] = 0.; s_y[t] = 0.; s_z[t] = 0.;
            s_o[t] = 1.; s_k[t] = 1.; s_w[t] = 0.;
            s_a[t] = -2;
        }
        __syncthreads();
        const bool diag = J == I;
        for (int c = 0; c < VV10_WARPS; c++) {
            double Tjx = 0, Tjy = 0, Tjz = 0;
            if (!diag || c >= warp) {
                for (int s = 0; s < 32; s++) {
                    const int jj = c * 32 + (lane + s) % 32;
                    const double dx = r_i.x - s_x[jj];
                    const double dy = r_i.y - s_y[jj];
                    const double dz = r_i.z - s_z[jj];
                    const double omega_j = s_o[jj];
                    const double kappa_j = s_k[jj];
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i * r_ij2 + kappa_i;
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_sum = g_ij + g_ji;
                    const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
                    const double Phi_ij = -g_ij_ji_sum_1;
                    const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
                    // pairs of points on different atoms; in a diagonal tile j > i (j == i is on one atom anyway)
                    if (s_a[jj] != atom_i && (!diag || jj > t)) {
                        const double K = rho_weight_i * s_w[jj] * Phi_ij * dPhi_drj_over_Phi;
                        Tix -= K * dx;
                        Tiy -= K * dy;
                        Tiz -= K * dz;
                        Tjx += K * dx;
                        Tjy += K * dy;
                        Tjz += K * dz;
                    }
                    Tjx = __shfl_sync(0xffffffff, Tjx, next);
                    Tjy = __shfl_sync(0xffffffff, Tjy, next);
                    Tjz = __shfl_sync(0xffffffff, Tjz, next);
                }
            }
            s_acc[warp][0][c * 32 + lane] = Tjx;
            s_acc[warp][1][c * 32 + lane] = Tjy;
            s_acc[warp][2][c * 32 + lane] = Tjz;
        }
        __syncthreads();
        if (jt < ngrids) {
            double tx = 0, ty = 0, tz = 0;
            for (int k = 0; k < VV10_WARPS; k++) {
                tx += s_acc[k][0][t];
                ty += s_acc[k][1][t];
                tz += s_acc[k][2][t];
            }
            atomicAdd(T + 0 * ngrids + jt, tx);
            atomicAdd(T + 1 * ngrids + jt, ty);
            atomicAdd(T + 2 * ngrids + jt, tz);
        }
        __syncthreads();
    }
    if (i < ngrids) {
        atomicAdd(T + 0 * ngrids + i, Tix);
        atomicAdd(T + 1 * ngrids + i, Tiy);
        atomicAdd(T + 2 * ngrids + i, Tiz);
    }
}

extern "C" {
// VV10 gradient, the grid response's pair term (part 28): T (3 x ngrids, zeroed here) with T_j = sum over the
// points i on other atoms of K_ij (r_i - r_j); the term for atom B is 3 sum_{j on B} T_j
__host__
int VXC_vv10nlc_grad_grid_response_T(const cudaStream_t stream, double* T, const double* grid_coord,
                                     const double* rho_weight, const double* omega, const double* kappa,
                                     const int* atom, const int ngrids)
{
    if (ngrids <= 0) {
        return 0;
    }
    cudaMemsetAsync(T, 0, 3 * ngrids * sizeof(double), stream);
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const dim3 blocks(ntiles, (ntiles + VV10_JTILES - 1) / VV10_JTILES);
    vv10_grad_grid_response_T_kernel<<<blocks, VV10_TILE, 0, stream>>>(T, grid_coord, rho_weight, omega, kappa,
                                                                       atom, ngrids, VV10_JTILES);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad grid response T: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_fock_eval_UWE(const cudaStream_t stream,
                              double* U, double* W, double* E,
                              const double* grid_coord, const double* rho_weight,
                              const double* omega, const double* kappa,
                              const int ngrids)
{
    // each pair of points once (part 26): the kernel sums into U, W and E, vv10_scale_UWE_kernel applies the factors
    if (ngrids <= 0) {
        return 0;
    }
    cudaMemsetAsync(U, 0, ngrids * sizeof(double), stream);
    cudaMemsetAsync(W, 0, ngrids * sizeof(double), stream);
    cudaMemsetAsync(E, 0, ngrids * sizeof(double), stream);
    const int ntiles = (ngrids + VV10_TILE - 1) / VV10_TILE;
    const dim3 blocks(ntiles, (ntiles + VV10_JTILES - 1) / VV10_JTILES);
    vv10_fock_eval_UWE_rot_kernel<<<blocks, VV10_TILE, 0, stream>>>(U, W, E, grid_coord, rho_weight, omega, kappa,
                                                                    ngrids, VV10_JTILES);
    vv10_scale_UWE_kernel<<<(ngrids + 255) / 256, 256, 0, stream>>>(U, W, E, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_UWE: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_fock_eval_omega_derivative(const cudaStream_t stream,
                                           double* omega, double* domega_drho, double* domega_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_fock_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(omega, domega_drho, domega_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(const cudaStream_t stream,
                                                      double* Egr,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                      const int natm, const int i_grid_begin, const int ngrids)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_grad_eval_E_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        Egr, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, natm, i_grid_begin, ngrids
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad eval_E_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_UWABCE(const cudaStream_t stream,
                                 double* U, double* W, double* A, double* B, double* C, double* E,
                                 const double* grid_coord, const double* rho_weight,
                                 const double* omega, const double* kappa,
                                 const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_UWABCE_kernel<<<blocks, threads, 0, stream>>>(U, W, A, B, C, E,
                                                                 grid_coord, rho_weight, omega, kappa, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_UWABCE: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_omega_derivative(const cudaStream_t stream,
                                           double* omega, double* domega_drho, double* domega_dgamma,
                                           double* d2omega_drho2, double* d2omega_dgamma2, double* d2omega_drho_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(omega, domega_drho, domega_dgamma,
                                                                           d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_f_t(const cudaStream_t stream,
                              double* f_rho_t, double* f_gamma_t,
                              const double* grid_coord, const double* grid_weight,
                              const double* rho, const double* omega, const double* kappa,
                              const double* U, const double* W, const double* A, const double* B, const double* C,
                              const double* domega_drho, const double* domega_dgamma, const double* dkappa_drho,
                              const double* d2omega_drho2, const double* d2omega_dgamma2, const double* d2omega_drho_dgamma, const double* d2kappa_drho2,
                              const double* rho_t, const double* gamma_t,
                              const int ngrids, const int ntrial)
{
    {
        constexpr int n_trial_per_thread = 6; // Notice: ntrial is likely a multiple of 3
        const dim3 threads(NG_PER_BLOCK, 1);
        const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                        (ntrial + n_trial_per_thread - 1) / n_trial_per_thread);
        vv10_hess_eval_f_t_offdiagonal_kernel<n_trial_per_thread> <<<blocks, threads, 0, stream>>> (
            f_rho_t, f_gamma_t,
            grid_coord, grid_weight, rho, omega, kappa,
            domega_drho, domega_dgamma, dkappa_drho,
            rho_t, gamma_t, ngrids, ntrial
        );
    }
    {
        const dim3 threads(NG_PER_BLOCK);
        const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK);
        vv10_hess_eval_f_t_diagonal_kernel<<<blocks, threads, 0, stream>>> (
            f_rho_t, f_gamma_t,
            rho,
            U, W, A, B, C,
            domega_drho, domega_dgamma, dkappa_drho,
            d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2,
            rho_t, gamma_t, ngrids, ntrial
        );
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_f_t: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_grid_response_offdiagonal(const cudaStream_t stream,
                                                        double* Egr, double* Ugr, double* Wgr,
                                                        const double* grid_coord, const double* rho_weight,
                                                        const double* omega, const double* kappa,
                                                        const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                        const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_hess_eval_EUW_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        Egr, Ugr, Wgr, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, ngrids, natm);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_with_weight1(const cudaStream_t stream,
                                           double* Ew, double* Uw, double* Ww,
                                           const double* grid_coord, const double* grid_weight1,
                                           const double* rho, const double* omega, const double* kappa,
                                           const int ngrids, const int nderivative)
{
    constexpr int n_derivative_per_thread = 6; // Notice: ntrial is always a multiple of 3
    const dim3 threads(NG_PER_BLOCK, 1);
    const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                      (nderivative + n_derivative_per_thread - 1) / n_derivative_per_thread);
    vv10_hess_eval_EUW_with_weight1_kernel<n_derivative_per_thread> <<<blocks, threads, 0, stream>>> (
        Ew, Uw, Ww,
        grid_coord, grid_weight1, rho, omega, kappa,
        ngrids, nderivative
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_with_weight1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_D_B_in_double_grid_response_offdiagonal(const cudaStream_t stream,
                                                      double* D_B,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                      const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_hess_eval_D_B_in_double_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        D_B, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, ngrids, natm
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_E_grgr_AB: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
