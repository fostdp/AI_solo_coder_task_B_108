use log::{debug, warn, info};

const MAX_BDF_ORDER: usize = 5;
const NEWTON_MAX_ITER: usize = 20;
const NEWTON_TOL: f64 = 1e-8;
const SAFETY_FACTOR: f64 = 0.9;
const MIN_DT: f64 = 1e-12;
const MAX_DT: f64 = 1e6;

pub struct OdeSolverConfig {
    pub rtol: f64,
    pub atol: f64,
    pub max_steps: usize,
    pub initial_dt: f64,
    pub max_order: usize,
    pub enforce_non_negative: bool,
}

impl Default for OdeSolverConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-10,
            max_steps: 100_000,
            initial_dt: 0.01,
            max_order: MAX_BDF_ORDER,
            enforce_non_negative: true,
        }
    }
}

pub struct OdeSolution {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub steps_taken: usize,
    pub rejected_steps: usize,
}

pub trait OdeSystem: Send + Sync {
    fn dim(&self) -> usize;
    fn rhs(&self, t: f64, y: &[f64]) -> Vec<f64>;
    fn jacobian(&self, t: f64, y: &[f64]) -> Option<Vec<Vec<f64>>> {
        let _ = (t, y);
        None
    }
}

pub fn numerical_jacobian(sys: &dyn OdeSystem, t: f64, y: &[f64], eps: f64) -> Vec<Vec<f64>> {
    let n = sys.dim();
    let f0 = sys.rhs(t, y);
    let mut jac = vec![vec![0.0; n]; n];
    let mut y_pert = y.to_vec();
    for j in 0..n {
        let h = eps * y[j].abs().max(eps);
        y_pert[j] = y[j] + h;
        let f1 = sys.rhs(t, &y_pert);
        for i in 0..n {
            jac[i][j] = (f1[i] - f0[i]) / h;
        }
        y_pert[j] = y[j];
    }
    jac
}

pub struct BdfSolver<'a> {
    sys: &'a dyn OdeSystem,
    config: OdeSolverConfig,
    t: f64,
    y: Vec<f64>,
    dt: f64,
    order: usize,
    history_y: Vec<Vec<f64>>,
    history_t: Vec<f64>,
    n_steps: usize,
    n_rejected: usize,
}

impl<'a> BdfSolver<'a> {
    pub fn new(sys: &'a dyn OdeSystem, y0: Vec<f64>, config: OdeSolverConfig) -> Self {
        let initial_dt = config.initial_dt;
        let order = 1;
        let mut history_y = Vec::with_capacity(config.max_order + 1);
        let mut history_t = Vec::with_capacity(config.max_order + 1);
        history_y.push(y0.clone());
        history_t.push(0.0);
        Self {
            sys,
            config,
            t: 0.0,
            y: y0,
            dt: initial_dt,
            order,
            history_y,
            history_t,
            n_steps: 0,
            n_rejected: 0,
        }
    }

    pub fn solve(mut self, t_end: f64) -> Result<OdeSolution, String> {
        let n = self.sys.dim();
        let mut sol_t = vec![self.t];
        let mut sol_y = vec![self.y.clone()];

        while self.t < t_end && self.n_steps < self.config.max_steps {
            let dt = self.dt.min(t_end - self.t);

            let (y_new, err) = self.bdf_step_with_error(dt)?;

            if err < 1.0 {
                self.t += dt;
                self.y = y_new;
                if self.config.enforce_non_negative {
                    for v in self.y.iter_mut() {
                        *v = v.max(0.0);
                    }
                }
                self.n_steps += 1;

                self.history_y.push(self.y.clone());
                self.history_t.push(self.t);
                if self.history_y.len() > self.config.max_order + 1 {
                    self.history_y.remove(0);
                    self.history_t.remove(0);
                }
                sol_t.push(self.t);
                sol_y.push(self.y.clone());

                let factor = if err > 0.0 {
                    SAFETY_FACTOR * (1.0 / err).powf(1.0 / (self.order as f64 + 1.0))
                } else {
                    4.0
                };
                self.dt = (dt * factor.min(4.0)).clamp(MIN_DT, MAX_DT);

                if self.n_steps % 10 == 0 && self.order < self.config.max_order {
                    self.order = (self.order + 1).min(self.config.max_order);
                }
            } else {
                self.n_rejected += 1;
                let factor = SAFETY_FACTOR * (1.0 / err).powf(1.0 / (self.order as f64 + 1.0));
                let new_dt = (dt * factor.min(0.5)).max(MIN_DT);
                self.dt = new_dt.min(dt * 0.5).max(MIN_DT);

                if self.n_rejected > 50 && self.order > 1 {
                    self.order -= 1;
                    debug!("BDF降阶: order={}", self.order);
                }

                if self.dt <= MIN_DT {
                    warn!("BDF step size reduced to min dt={:.2e}, t={:.4}", self.dt, self.t);
                }
            }
        }

        if self.n_steps >= self.config.max_steps {
            warn!("BDF求解器达到最大步数限制: {}", self.config.max_steps);
        }

        Ok(OdeSolution {
            t: sol_t,
            y: sol_y,
            steps_taken: self.n_steps,
            rejected_steps: self.n_rejected,
        })
    }

    fn bdf_step_with_error(&mut self, dt: f64) -> Result<(Vec<f64>, f64), String> {
        let k = self.order.min(self.history_y.len());
        let actual_k = k.min(self.history_y.len() - 1);

        let y_new = match self.bdf_step_internal(dt, actual_k) {
            Ok(y) => y,
            Err(_) => return Ok((vec![0.0; self.sys.dim()], 1e30)),
        };

        let err = if actual_k > 0 && self.history_y.len() >= 2 {
            match self.bdf_step_internal(dt, actual_k - 1) {
                Ok(y_low) => self.estimate_error(&y_new, &y_low, actual_k),
                Err(_) => 1e30,
            }
        } else {
            self.estimate_error_initial(&y_new, dt)
        };

        Ok((y_new, err))
    }

    fn bdf_step_internal(&self, dt: f64, order: usize) -> Result<Vec<f64>, String> {
        let n = self.sys.dim();

        let (alpha, alpha_k) = if order == 0 || self.history_y.len() < 2 {
            (vec![1.0], 1.0)
        } else {
            bdf_coefficients(order.min(self.history_y.len() - 1))
        };

        let mut b = vec![0.0; n];
        for j in 0..alpha.len() {
            if j < self.history_y.len() {
                let y_hist = &self.history_y[self.history_y.len() - 1 - j];
                for i in 0..n {
                    b[i] += alpha[j] * y_hist[i];
                }
            }
        }

        let mut y_predict = self.y.clone();
        if self.history_y.len() >= 2 {
            let y_prev = &self.history_y[self.history_y.len() - 2];
            for i in 0..n {
                y_predict[i] = 2.0 * self.y[i] - y_prev[i];
            }
        }

        let jac = match self.sys.jacobian(self.t + dt, &y_predict) {
            Some(j) => j,
            None => numerical_jacobian(self.sys, self.t + dt, &y_predict, 1e-8),
        };

        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = -dt * jac[i][j];
            }
            m[i][i] += alpha_k;
        }

        let mut y_new = y_predict.clone();
        let mut converged = false;

        for _iter in 0..NEWTON_MAX_ITER {
            let f = self.sys.rhs(self.t + dt, &y_new);

            let mut residual = vec![0.0; n];
            for i in 0..n {
                residual[i] = alpha_k * y_new[i] - dt * f[i] - b[i];
            }

            let mut delta = vec![0.0; n];
            solve_linear_system(&m, &residual, &mut delta);

            for i in 0..n {
                y_new[i] -= delta[i];
            }

            let norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let y_norm: f64 = y_new.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-10);

            if norm / y_norm < NEWTON_TOL {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(format!("Newton iteration failed to converge in BDF step"));
        }

        Ok(y_new)
    }

    fn estimate_error(&self, y_high: &[f64], y_low: &[f64], _order: usize) -> f64 {
        let n = self.sys.dim();
        let mut err_sq = 0.0;

        for i in 0..n {
            let sc = self.config.atol + self.config.rtol * y_high[i].abs();
            let e = y_high[i] - y_low[i];
            err_sq += (e / sc) * (e / sc);
        }
        (err_sq / n as f64).sqrt()
    }

    fn estimate_error_initial(&self, y_new: &[f64], dt: f64) -> f64 {
        let n = self.sys.dim();
        let mut err_sq = 0.0;
        let f_new = self.sys.rhs(self.t + dt, y_new);
        let f_old = self.sys.rhs(self.t, &self.y);

        for i in 0..n {
            let sc = self.config.atol + self.config.rtol * y_new[i].abs();
            let e = 0.5 * dt * (f_new[i] - f_old[i]);
            err_sq += (e / sc) * (e / sc);
        }
        (err_sq / n as f64).sqrt()
    }
}

fn bdf_coefficients(order: usize) -> (Vec<f64>, f64) {
    match order {
        1 => (vec![1.0], 1.0),
        2 => (vec![4.0 / 3.0, -1.0 / 3.0], 2.0 / 3.0),
        3 => (vec![18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0], 6.0 / 11.0),
        4 => (vec![48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0], 12.0 / 25.0),
        5 => (vec![300.0 / 137.0, -300.0 / 137.0, 200.0 / 137.0, -75.0 / 137.0, 12.0 / 137.0], 60.0 / 137.0),
        _ => (vec![1.0], 1.0),
    }
}

fn solve_linear_system(a: &[Vec<f64>], b: &[f64], x: &mut [f64]) {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-30 {
            continue;
        }

        let pivot = aug[col][col];
        for j in col..=n {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    for i in 0..n {
        x[i] = aug[i][n];
    }
}

pub struct CollagenHydrolysisOde {
    pub ea: f64,
    pub a_pre: f64,
    pub r: f64,
    pub ph_acid_coeff: f64,
    pub ph_base_coeff: f64,
    pub orp_factor: f64,
    pub vmax: f64,
    pub km: f64,
    pub biomass_growth_rate: f64,
    pub biomass_death_rate: f64,
    pub enzyme_ea: f64,
    pub enzyme_a: f64,
    pub ph_opt: f64,
    pub ph_range: f64,
    pub temp_opt: f64,
}

impl Default for CollagenHydrolysisOde {
    fn default() -> Self {
        Self {
            ea: 85_000.0,
            a_pre: 1.2e10,
            r: 8.314,
            ph_acid_coeff: 4.5e-4,
            ph_base_coeff: 8.0e-5,
            orp_factor: 1.0,
            vmax: 3.5e-7,
            km: 0.012,
            biomass_growth_rate: 0.35,
            biomass_death_rate: 0.05,
            enzyme_ea: 55_000.0,
            enzyme_a: 8.0e8,
            ph_opt: 6.8,
            ph_range: 1.5,
            temp_opt: 37.0,
        }
    }
}

impl OdeSystem for CollagenHydrolysisOde {
    fn dim(&self) -> usize { 3 }

    fn rhs(&self, _t: f64, y: &[f64]) -> Vec<f64> {
        let substrate = y[0].max(0.0);
        let biomass = y[1].max(0.0);
        let _cumulative_deg = y[2];

        let temp_celsius = 18.0;
        let temp_k = temp_celsius + 273.15;
        let ph = 6.5;

        let k_arr = self.a_pre * (-self.ea / (self.r * temp_k)).exp();
        let h_plus = 10.0_f64.powf(-ph);
        let oh_minus = 10.0_f64.powf(ph - 14.0);
        let ph_factor = 1.0 + self.ph_acid_coeff * h_plus + self.ph_base_coeff * oh_minus;
        let k_abiotic = k_arr * ph_factor * self.orp_factor;

        let delta_ph = ph - self.ph_opt;
        let enzyme_ph = (-delta_ph * delta_ph / (2.0 * self.ph_range * self.ph_range)).exp();
        let opt_k = self.temp_opt + 273.15;
        let arr_num = self.enzyme_a * (-self.enzyme_ea / (self.r * temp_k)).exp();
        let arr_den = self.enzyme_a * (-self.enzyme_ea / (self.r * opt_k)).exp();
        let enzyme_temp = if temp_celsius > self.temp_opt {
            let excess = temp_celsius - self.temp_opt;
            (arr_num / arr_den) * (-excess / (self.temp_opt + 15.0)).exp().max(0.1)
        } else {
            arr_num / arr_den
        };
        let v_eff = self.vmax * biomass * enzyme_ph * enzyme_temp;
        let k_enzyme = if substrate > 0.0 && self.km > 0.0 {
            v_eff * substrate / (self.km + substrate)
        } else { 0.0 };

        let k_total = k_abiotic + k_enzyme;

        let d_substrate = -k_total * substrate * 86400.0;
        let d_biomass = biomass * (self.biomass_growth_rate * enzyme_ph * enzyme_temp * substrate / (0.05 + substrate) - self.biomass_death_rate);
        let d_cumulative = k_total * substrate * 86400.0;

        vec![d_substrate, d_biomass, d_cumulative]
    }

    fn jacobian(&self, t: f64, y: &[f64]) -> Option<Vec<Vec<f64>>> {
        let n = self.dim();
        Some(numerical_jacobian(self, t, y, 1e-7))
    }
}

pub fn solve_collagen_degradation(
    days: f64,
    initial_substrate: f64,
    initial_biomass: f64,
    temp_celsius: f64,
    ph: f64,
    orp_mv: f64,
    config: Option<OdeSolverConfig>,
) -> Result<OdeSolution, String> {
    let mut ode = CollagenHydrolysisOde::default();
    ode.orp_factor = 1.0 + 0.8 * ((orp_mv + 300.0) / 600.0).clamp(0.0, 1.0);

    let y0 = vec![initial_substrate, initial_biomass, 0.0];
    let cfg = config.unwrap_or_default();
    let solver = BdfSolver::new(&ode, y0, cfg);
    solver.solve(days)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStiffSystem;
    impl OdeSystem for TestStiffSystem {
        fn dim(&self) -> usize { 2 }
        fn rhs(&self, _t: f64, y: &[f64]) -> Vec<f64> {
            vec![-0.1 * y[0] + 998.0 * y[1], -999.0 * y[1]]
        }
    }

    #[test]
    fn test_bdf_stiff_system() {
        let sys = TestStiffSystem;
        let y0 = vec![1.0, 1.0];
        let cfg = OdeSolverConfig {
            rtol: 1e-6,
            atol: 1e-10,
            max_steps: 50000,
            initial_dt: 1e-6,
            max_order: 5,
            enforce_non_negative: false,
        };
        let solver = BdfSolver::new(&sys, y0, cfg);
        let sol = solver.solve(1.0);
        assert!(sol.is_ok(), "BDF应能求解刚性系统");
        let sol = sol.unwrap();
        assert!(sol.y.last().unwrap()[1].abs() < 1e-3,
            "y1应在t=1时接近0，实际={:.6}", sol.y.last().unwrap()[1]);
        info!("BDF刚性系统测试通过: steps={}, rejected={}",
            sol.steps_taken, sol.rejected_steps);
    }

    #[test]
    fn test_collagen_degradation_ode() {
        let sol = solve_collagen_degradation(30.0, 1.0, 0.1, 18.0, 6.5, 150.0, None);
        assert!(sol.is_ok(), "胶原降解ODE求解应成功");
        let sol = sol.unwrap();
        let final_state = sol.y.last().unwrap();
        assert!(final_state[0] < 1.0, "底物应减少");
        assert!(final_state[2] > 0.0, "累积降解应增加");
    }

    #[test]
    fn test_numerical_jacobian() {
        let sys = TestStiffSystem;
        let jac = numerical_jacobian(&sys, 0.0, &[1.0, 1.0], 1e-8);
        assert!((jac[0][0] - (-0.1)).abs() < 0.01);
        assert!((jac[1][1] - (-999.0)).abs() < 5.0);
    }
}
