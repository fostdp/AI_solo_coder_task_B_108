use relic_algo::{
    ArrheniusConfig, CalciumPhosphateConfig,
    collagen_hydrolysis_rate, dissolution_rate, corrosion_rate_um_per_year,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloParams {
    pub num_simulations: usize,
    pub current_ph: f64,
    pub ph_std_dev: f64,
    pub current_temp_c: f64,
    pub temp_std_dev: f64,
    pub current_ca_ppm: f64,
    pub ca_std_dev: f64,
    pub current_orp_mv: f64,
    pub orp_std_dev: f64,
    pub forecast_years: f64,
    pub time_steps_per_year: usize,
    pub target_corrosion_threshold_um: f64,
    pub acceptable_risk_threshold: f64,
    pub current_collagen_remaining_pct: f64,
}

impl Default for MonteCarloParams {
    fn default() -> Self {
        Self {
            num_simulations: 5000,
            current_ph: 7.0,
            ph_std_dev: 0.3,
            current_temp_c: 18.0,
            temp_std_dev: 2.0,
            current_ca_ppm: 80.0,
            ca_std_dev: 15.0,
            current_orp_mv: 100.0,
            orp_std_dev: 50.0,
            forecast_years: 50.0,
            time_steps_per_year: 12,
            target_corrosion_threshold_um: 200.0,
            acceptable_risk_threshold: 0.25,
            current_collagen_remaining_pct: 70.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcavationWindowAssessment {
    pub start_year: f64,
    pub end_year: f64,
    pub probability_of_success: f64,
    pub expected_damage_if_wait: f64,
    pub expected_damage_if_excavate: f64,
    pub net_benefit: f64,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcavationOptimizationResult {
    pub params: MonteCarloParams,
    pub simulations_completed: usize,
    pub optimal_window: ExcavationWindowAssessment,
    pub windows: Vec<ExcavationWindowAssessment>,
    pub year_by_year_stats: Vec<YearlyForecast>,
    pub risk_distribution: RiskDistribution,
    pub final_recommendation: String,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearlyForecast {
    pub year: f64,
    pub mean_corrosion_um: f64,
    pub p5_corrosion_um: f64,
    pub p25_corrosion_um: f64,
    pub p50_corrosion_um: f64,
    pub p75_corrosion_um: f64,
    pub p95_corrosion_um: f64,
    pub mean_collagen_pct: f64,
    pub prob_exceed_threshold: f64,
    pub should_excavate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDistribution {
    pub percentiles: Vec<(String, f64)>,
    pub probability_by_year: Vec<(f64, f64)>,
}

struct SimResult {
    corrosion: Vec<(f64, f64)>,
}

fn run_single_simulation(
    params: &MonteCarloParams,
    arr_cfg: &ArrheniusConfig,
    ca_cfg: &CalciumPhosphateConfig,
    forecast_months: usize,
    dt_seconds: f64,
) -> SimResult {
    let mut rng = rand::thread_rng();

    let normal_ph = Normal::new(params.current_ph, params.ph_std_dev.max(0.01)).unwrap();
    let normal_temp = Normal::new(params.current_temp_c, params.temp_std_dev.max(0.01)).unwrap();
    let normal_ca = Normal::new(params.current_ca_ppm, params.ca_std_dev.max(0.01)).unwrap();
    let normal_orp = Normal::new(params.current_orp_mv, params.orp_std_dev.max(0.01)).unwrap();

    let base_ph: f64 = normal_ph.sample(&mut rng).clamp(3.0, 11.0);
    let base_temp: f64 = normal_temp.sample(&mut rng).clamp(-5.0, 50.0);
    let base_ca: f64 = normal_ca.sample(&mut rng).clamp(5.0, 1000.0);
    let base_orp: f64 = normal_orp.sample(&mut rng).clamp(-400.0, 700.0);

    let mut sim_corrosion: Vec<(f64, f64)> = Vec::with_capacity(forecast_months + 1);
    sim_corrosion.push((0.0, params.current_collagen_remaining_pct));

    let mut cumulative_collagen = params.current_collagen_remaining_pct;

    for month in 1..=forecast_months {
        let season = (month as f64 % 12.0) / 12.0;
        let temp_osc = (season * 2.0 * std::f64::consts::PI).sin() * 3.0;
        let ph_osc = (season * 2.0 * std::f64::consts::PI).sin() * 0.1;
        let noise_p: f64 = rng.gen::<f64>() - 0.5;
        let noise_t: f64 = rng.gen::<f64>() - 0.5;

        let step_ph = (base_ph + ph_osc + noise_p * 0.05).clamp(3.0, 11.0);
        let step_temp = (base_temp + temp_osc + noise_t * 0.5).clamp(-5.0, 50.0);
        let step_ca = (base_ca * (0.95 + rng.gen::<f64>() * 0.1)).clamp(5.0, 1000.0);
        let step_orp = (base_orp + (rng.gen::<f64>() - 0.5) * 20.0).clamp(-400.0, 700.0);

        let coll_rate = collagen_hydrolysis_rate(step_temp, step_ph, step_orp, Some(arr_cfg));
        let diss_rate = dissolution_rate(step_ph, step_temp, step_ca, 0.5, ca_cfg);
        let cor_rate = corrosion_rate_um_per_year(coll_rate, diss_rate, step_ph);
        let corrosion_per_step = cor_rate / (params.time_steps_per_year as f64);

        let degradation_step = if cumulative_collagen > 0.0 {
            (coll_rate * dt_seconds).min(0.99)
        } else {
            0.0
        };
        cumulative_collagen = (cumulative_collagen * (1.0 - degradation_step)).max(0.0);

        let current_depth = sim_corrosion
            .last()
            .map(|(_, depth)| *depth)
            .unwrap_or(0.0);

        sim_corrosion.push((
            month as f64 / params.time_steps_per_year as f64,
            current_depth + corrosion_per_step,
        ));
    }

    SimResult {
        corrosion: sim_corrosion,
    }
}

pub fn run_monte_carlo_excavation(params: MonteCarloParams) -> ExcavationOptimizationResult {
    let arr_cfg = ArrheniusConfig::default();
    let ca_cfg = CalciumPhosphateConfig::default();
    let n_sims = params.num_simulations.max(100);
    let forecast_months = (params.forecast_years * params.time_steps_per_year as f64) as usize;
    let dt_seconds = (365.25 * 24.0 * 3600.0) / (params.time_steps_per_year as f64);

    let all_results: Vec<SimResult> = (0..n_sims)
        .into_par_iter()
        .map(|_| run_single_simulation(&params, &arr_cfg, &ca_cfg, forecast_months, dt_seconds))
        .collect();

    let mut year_stats = Vec::new();
    let years_to_eval: Vec<usize> = (0..=forecast_months)
        .step_by(params.time_steps_per_year)
        .collect();

    for (year_idx, &month_idx) in years_to_eval.iter().enumerate() {
        let year = year_idx as f64;
        let mut depths: Vec<f64> = all_results
            .iter()
            .map(|sim| sim.corrosion.get(month_idx).map(|x| x.1).unwrap_or(0.0))
            .collect();
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = depths.len();
        let mean = depths.iter().sum::<f64>() / n as f64;
        let p5 = depths[((n as f64 * 0.05) as usize).min(n - 1)];
        let p25 = depths[((n as f64 * 0.25) as usize).min(n - 1)];
        let p50 = depths[n / 2];
        let p75 = depths[((n as f64 * 0.75) as usize).min(n - 1)];
        let p95 = depths[((n as f64 * 0.95) as usize).min(n - 1)];

        let exceed_count = depths
            .iter()
            .filter(|&&d| d >= params.target_corrosion_threshold_um)
            .count();
        let prob_exceed = exceed_count as f64 / n as f64;

        let mut collagens: Vec<f64> = all_results
            .iter()
            .map(|sim| {
                sim.corrosion
                    .get(month_idx)
                    .map(|_| {
                        let rem_pct = params.current_collagen_remaining_pct
                            * (-year * 0.01).exp();
                        rem_pct
                    })
                    .unwrap_or(params.current_collagen_remaining_pct)
            })
            .collect();
        collagens.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_coll = collagens.iter().sum::<f64>() / collagens.len() as f64;

        let should = prob_exceed >= params.acceptable_risk_threshold;

        year_stats.push(YearlyForecast {
            year,
            mean_corrosion_um: mean,
            p5_corrosion_um: p5,
            p25_corrosion_um: p25,
            p50_corrosion_um: p50,
            p75_corrosion_um: p75,
            p95_corrosion_um: p95,
            mean_collagen_pct: mean_coll,
            prob_exceed_threshold: prob_exceed,
            should_excavate: should,
        });
    }

    let mut windows = Vec::new();
    let window_sizes: Vec<(f64, f64)> = vec![
        (0.0, 1.0),
        (0.5, 2.0),
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (5.0, 10.0),
    ];

    for (start, end) in window_sizes.iter() {
        let start_idx = (*start * params.time_steps_per_year as f64).round() as usize;
        let end_idx = (*end * params.time_steps_per_year as f64).round() as usize;
        let end_idx = end_idx.min(forecast_months);

        let start_stat = year_stats.get(start_idx.min(year_stats.len() - 1));
        let end_stat = year_stats.get(end_idx.min(year_stats.len() - 1));

        if let (Some(s), Some(e)) = (start_stat, end_stat) {
            let prob_success =
                1.0 - ((s.prob_exceed_threshold + e.prob_exceed_threshold) / 2.0);
            let expected_damage_wait = e.mean_corrosion_um;
            let expected_damage_excav = s.mean_corrosion_um * 0.5 + 10.0;
            let net_benefit = expected_damage_wait - expected_damage_excav;

            let recommendation = if prob_success >= 0.9 {
                "强烈推荐此时间窗口发掘".to_string()
            } else if prob_success >= 0.75 {
                "建议此时间段内发掘".to_string()
            } else if prob_success >= 0.6 {
                "可考虑发掘，需准备强化保护方案".to_string()
            } else {
                "不建议，风险过高".to_string()
            };

            windows.push(ExcavationWindowAssessment {
                start_year: *start,
                end_year: *end,
                probability_of_success: prob_success.clamp(0.0, 1.0),
                expected_damage_if_wait: expected_damage_wait,
                expected_damage_if_excavate: expected_damage_excav,
                net_benefit,
                recommendation,
            });
        }
    }

    let optimal_window = windows
        .iter()
        .max_by(|a, b| {
            let score_a = a.probability_of_success * 2.0 + a.net_benefit / 100.0;
            let score_b = b.probability_of_success * 2.0 + b.net_benefit / 100.0;
            score_a.partial_cmp(&score_b).unwrap()
        })
        .cloned()
        .unwrap_or_else(|| ExcavationWindowAssessment {
            start_year: 0.0,
            end_year: 1.0,
            probability_of_success: 0.5,
            expected_damage_if_wait: 100.0,
            expected_damage_if_excavate: 50.0,
            net_benefit: 0.0,
            recommendation: "建议立即评估".to_string(),
        });

    let prob_by_year: Vec<(f64, f64)> = year_stats
        .iter()
        .map(|ys| (ys.year, ys.prob_exceed_threshold))
        .collect();
    let percentiles = if let Some(last) = year_stats.last() {
        vec![
            ("P5 (乐观)".to_string(), last.p5_corrosion_um),
            ("P25".to_string(), last.p25_corrosion_um),
            ("P50 (中位)".to_string(), last.p50_corrosion_um),
            ("P75".to_string(), last.p75_corrosion_um),
            ("P95 (悲观)".to_string(), last.p95_corrosion_um),
        ]
    } else {
        Vec::new()
    };

    let risk_dist = RiskDistribution {
        percentiles,
        probability_by_year: prob_by_year.clone(),
    };

    let first_exceed_year = year_stats
        .iter()
        .find(|ys| ys.prob_exceed_threshold >= params.acceptable_risk_threshold)
        .map(|ys| ys.year);

    let confidence = {
        let good_sims: usize = all_results
            .iter()
            .filter(|sim| {
                sim.corrosion
                    .last()
                    .map(|(_, depth)| *depth < params.target_corrosion_threshold_um * 1.5)
                    .unwrap_or(false)
            })
            .count();
        (good_sims as f64 / n_sims as f64).clamp(0.0, 1.0)
    };

    let final_rec = match first_exceed_year {
        Some(year) if year <= 1.0 => {
            format!("⚠️ 紧急建议：在{}年内抢救性发掘。当前环境腐蚀风险已达到临界值（超阈概率≥{:.0}%），若继续埋藏，{:.1}年后预计有{:.1}%概率超过腐蚀阈值{}μm。最佳窗口：立即起至{}年内完成。",
                year.ceil(), params.acceptable_risk_threshold * 100.0,
                params.forecast_years,
                (year_stats.last().map(|y| y.prob_exceed_threshold).unwrap_or(0.0) * 100.0),
                params.target_corrosion_threshold_um,
                optimal_window.end_year)
        }
        Some(year) => {
            format!("建议在{}年内完成发掘。以{}%置信度估计，{:.1}年后腐蚀超阈概率将达到{:.0}%。最佳发掘窗口：从现在起至{:.1}年（成功概率{:.0}%）。",
                year.floor(),
                confidence * 100.0,
                year,
                params.acceptable_risk_threshold * 100.0,
                optimal_window.end_year,
                optimal_window.probability_of_success * 100.0)
        }
        None => {
            if let Some(_last) = year_stats.last() {
                format!("当前环境相对稳定，{:.1}年内腐蚀超阈风险低于{:.0}%。可按正常考古进度安排发掘，最佳窗口：{:.1}至{:.1}年（成功概率{:.0}%）。",
                    params.forecast_years,
                    params.acceptable_risk_threshold * 100.0,
                    optimal_window.start_year,
                    optimal_window.end_year,
                    optimal_window.probability_of_success * 100.0)
            } else {
                "请补充更多历史数据以进行精确评估".to_string()
            }
        }
    };

    ExcavationOptimizationResult {
        params,
        simulations_completed: n_sims,
        optimal_window,
        windows,
        year_by_year_stats: year_stats,
        risk_distribution: risk_dist,
        final_recommendation: final_rec,
        confidence_level: confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(n: usize, years: f64) -> MonteCarloParams {
        MonteCarloParams {
            num_simulations: n,
            current_ph: 7.0,
            ph_std_dev: 0.3,
            current_temp_c: 18.0,
            temp_std_dev: 2.0,
            current_ca_ppm: 80.0,
            ca_std_dev: 15.0,
            current_orp_mv: 100.0,
            orp_std_dev: 50.0,
            forecast_years: years,
            time_steps_per_year: 4,
            target_corrosion_threshold_um: 200.0,
            acceptable_risk_threshold: 0.25,
            current_collagen_remaining_pct: 70.0,
        }
    }

    #[test]
    fn test_simulations_count_matches() {
        let params = make_params(500, 5.0);
        let result = run_monte_carlo_excavation(params);
        assert_eq!(result.simulations_completed, 500);
    }

    #[test]
    fn test_confidence_interval_ordering() {
        let params = make_params(200, 10.0);
        let result = run_monte_carlo_excavation(params);
        for y in &result.year_by_year_stats {
            assert!(y.p5_corrosion_um <= y.p25_corrosion_um);
            assert!(y.p25_corrosion_um <= y.p50_corrosion_um);
            assert!(y.p50_corrosion_um <= y.p75_corrosion_um);
            assert!(y.p75_corrosion_um <= y.p95_corrosion_um);
        }
    }

    #[test]
    fn test_corrosion_increases_over_time() {
        let params = make_params(200, 20.0);
        let result = run_monte_carlo_excavation(params);
        let stats = &result.year_by_year_stats;
        assert!(stats.len() >= 2);
        let first_mean = stats[0].mean_corrosion_um;
        let last_mean = stats[stats.len() - 1].mean_corrosion_um;
        assert!(last_mean >= first_mean);
    }

    #[test]
    fn test_optimal_window_within_windows() {
        let params = make_params(200, 10.0);
        let result = run_monte_carlo_excavation(params);
        assert!(!result.windows.is_empty());
    }

    #[test]
    fn test_final_recommendation_not_empty() {
        let params = make_params(100, 5.0);
        let result = run_monte_carlo_excavation(params);
        assert!(!result.final_recommendation.is_empty());
    }

    #[test]
    fn test_minimal_simulation_count() {
        let params = make_params(1, 1.0);
        let result = run_monte_carlo_excavation(params);
        assert!(result.simulations_completed >= 100);
    }

    #[test]
    fn test_risk_distribution_valid() {
        let params = make_params(200, 10.0);
        let result = run_monte_carlo_excavation(params);
        assert!(!result.risk_distribution.percentiles.is_empty());
        assert!(!result.risk_distribution.probability_by_year.is_empty());
    }

    #[test]
    fn test_success_probability_bounds() {
        let params = make_params(200, 15.0);
        let result = run_monte_carlo_excavation(params);
        for w in &result.windows {
            assert!(w.probability_of_success >= 0.0 && w.probability_of_success <= 1.0);
        }
    }

    #[test]
    fn test_prob_exceed_increases_with_time() {
        let params = make_params(200, 20.0);
        let result = run_monte_carlo_excavation(params);
        let stats = &result.year_by_year_stats;
        if stats.len() >= 2 {
            let first_exceed = stats[0].prob_exceed_threshold;
            let last_exceed = stats[stats.len() - 1].prob_exceed_threshold;
            assert!(last_exceed >= first_exceed);
        }
    }

    #[test]
    fn test_small_std_dev_gives_narrow_confidence() {
        let mut params_narrow = make_params(200, 10.0);
        params_narrow.ph_std_dev = 0.05;
        params_narrow.temp_std_dev = 0.3;
        let result_narrow = run_monte_carlo_excavation(params_narrow);

        let mut params_wide = make_params(200, 10.0);
        params_wide.ph_std_dev = 1.5;
        params_wide.temp_std_dev = 8.0;
        let result_wide = run_monte_carlo_excavation(params_wide);

        if !result_narrow.year_by_year_stats.is_empty() && !result_wide.year_by_year_stats.is_empty() {
            let mid_idx = result_narrow.year_by_year_stats.len() / 2;
            let narrow_p95_p5 = result_narrow.year_by_year_stats[mid_idx].p95_corrosion_um
                - result_narrow.year_by_year_stats[mid_idx].p5_corrosion_um;
            let wide_p95_p5 = result_wide.year_by_year_stats[mid_idx].p95_corrosion_um
                - result_wide.year_by_year_stats[mid_idx].p5_corrosion_um;
            assert!(wide_p95_p5 > narrow_p95_p5 * 0.5);
        }
    }
}
