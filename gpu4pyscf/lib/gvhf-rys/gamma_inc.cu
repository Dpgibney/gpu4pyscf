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

#include <float.h>
#define EPS_FLOAT64     DBL_EPSILON
#define SQRTPIE4        .886226925452758013
// 1/(n + 1/2): the series and the downward recursion multiply where they divided (my-pyscf part 25)
#define BOYS_NINV       96
static __constant__ double c_inv_half[BOYS_NINV] = {
    2.0, 0.6666666666666666, 0.4, 0.2857142857142857,
    0.2222222222222222, 0.18181818181818182, 0.15384615384615385, 0.13333333333333333,
    0.11764705882352941, 0.10526315789473684, 0.09523809523809523, 0.08695652173913043,
    0.08, 0.07407407407407407, 0.06896551724137931, 0.06451612903225806,
    0.06060606060606061, 0.05714285714285714, 0.05405405405405406, 0.05128205128205128,
    0.04878048780487805, 0.046511627906976744, 0.044444444444444446, 0.0425531914893617,
    0.04081632653061224, 0.0392156862745098, 0.03773584905660377, 0.03636363636363636,
    0.03508771929824561, 0.03389830508474576, 0.03278688524590164, 0.031746031746031744,
    0.03076923076923077, 0.029850746268656716, 0.028985507246376812, 0.028169014084507043,
    0.0273972602739726, 0.02666666666666667, 0.025974025974025976, 0.02531645569620253,
    0.024691358024691357, 0.024096385542168676, 0.023529411764705882, 0.022988505747126436,
    0.02247191011235955, 0.02197802197802198, 0.021505376344086023, 0.021052631578947368,
    0.020618556701030927, 0.020202020202020204, 0.019801980198019802, 0.019417475728155338,
    0.01904761904761905, 0.018691588785046728, 0.01834862385321101, 0.018018018018018018,
    0.017699115044247787, 0.017391304347826087, 0.017094017094017096, 0.01680672268907563,
    0.01652892561983471, 0.016260162601626018, 0.016, 0.015748031496062992,
    0.015503875968992248, 0.015267175572519083, 0.015037593984962405, 0.014814814814814815,
    0.014598540145985401, 0.014388489208633094, 0.014184397163120567, 0.013986013986013986,
    0.013793103448275862, 0.013605442176870748, 0.013422818791946308, 0.013245033112582781,
    0.013071895424836602, 0.012903225806451613, 0.012738853503184714, 0.012578616352201259,
    0.012422360248447204, 0.012269938650306749, 0.012121212121212121, 0.011976047904191617,
    0.011834319526627219, 0.011695906432748537, 0.011560693641618497, 0.011428571428571429,
    0.011299435028248588, 0.0111731843575419, 0.011049723756906077, 0.01092896174863388,
    0.010810810810810811, 0.0106951871657754, 0.010582010582010581, 0.010471204188481676
};
// erf(sqrt(t)) is exactly 1 in double precision from here on (erfc(7) = 4.2e-23)
#define ERF_ONE_T       49.

__device__
static void eval_gamma_inc_fn(double *f, double t, int m, int sq_id, int block_size)
{
    if (t < EPS_FLOAT64) {
        f[sq_id] = 1.;
        for (int i = 1; i <= m; i++) {
            f[sq_id + i*block_size] = 1./(2*i+1);
        }
    } else if (m > 0 && t < m*.5+.5) {
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        // the terms shrink at least twofold (t/(k+1/2) < 1/2), so k stays below m + 55
        for (int k = m+1; x > tol && k < BOYS_NINV; k++) {
            x *= t * c_inv_half[k];
            s += x;
        }
        double fval = s * c_inv_half[m];
        f[sq_id + m*block_size] = fval;
        for (int i = m-1; i >= 0; i--) {
            fval = (e + t * fval) * c_inv_half[i];
            f[sq_id + i*block_size] = fval;
        }
    } else {
        double tt = sqrt(t);
        double fval = SQRTPIE4 / tt;
        if (t < ERF_ONE_T) {
            fval *= erf(tt);
        }
        f[sq_id] = fval;
        if (m > 0) {
            double e = .5 * exp(-t);
            double b = 1. / t;
            double b1 = .5;
            for (int i = 1; i <= m; i++) {
                fval = b * (b1 * fval - e);
                f[sq_id + i*block_size] = fval;
                b1 += 1.;
            }
        }
    }
}
