use relic_algo::{
    ArrheniusConfig, MOLAR_GAS_R,
    arrhenius_rate_constant,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureHistoryPoint {
    pub years_bp: f64,
    pub temp_celsius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcavationThermalPerturbation {
    pub surface_temp_c: f64,
    pub exposure_duration_days: f64,
    pub thermal_shock_amplification_factor: f64,
}

impl Default for ExcavationThermalPerturbation {
    fn default() -> Self {
        ExcavationThermalPerturbation {
            surface_temp_c: 35.0,
            exposure_duration_days: 1.0,
            thermal_shock_amplification_factor: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollagenPreservationIndex {
    pub cpi_score: f64,
    pub cpi_grade: String,
    pub equivalent_years_at_20c: f64,
    pub equivalent_years_after_perturbation_at_20c: f64,
    pub remaining_collagen_pct: f64,
    pub remaining_after_perturbation_pct: f64,
    pub predicted_half_life_years: f64,
    pub initial_half_life_years: f64,
    pub temperature_history: Vec<TemperatureHistoryPoint>,
    pub activation_energy: f64,
    pub burial_years: f64,
    pub average_temp_c: f64,
    pub excavation_perturbation_applied: bool,
    pub perturbation_loss_pct: f64,
    pub interpretation: String,
}

pub fn calculate_cpi(
    activation_energy: f64,
    burial_years: f64,
    current_temp_c: f64,
    temp_history: Option<Vec<TemperatureHistoryPoint>>,
    initial_collagen_fraction: f64,
    perturbation: Option<ExcavationThermalPerturbation>,
) -> CollagenPreservationIndex {
    let arr_cfg = ArrheniusConfig {
        ea: activation_energy,
        a: 5.0e3,
        r: MOLAR_GAS_R,
        ph_acid_coeff: 4.5e-4,
        ph_base_coeff: 8.0e-5,
        ph_neutral_point: 7.0,
    };

    let history = temp_history.unwrap_or_else(|| {
        vec![
            TemperatureHistoryPoint {
                years_bp: burial_years,
                temp_celsius: current_temp_c - 5.0,
            },
            TemperatureHistoryPoint {
                years_bp: burial_years * 0.5,
                temp_celsius: current_temp_c - 2.0,
            },
            TemperatureHistoryPoint {
                years_bp: 0.0,
                temp_celsius: current_temp_c,
            },
        ]
    });

    let mut total_equivalent_years = 0.0;
    let mut weighted_temp_sum = 0.0;
    let mut total_weight = 0.0;

    for i in 0..history.len() {
        let curr = &history[i];
        let next_years = if i + 1 < history.len() {
            history[i + 1].years_bp
        } else {
            0.0
        };
        let period_duration = (curr.years_bp - next_years).abs().max(0.0);

        let k_current = arrhenius_rate_constant(curr.temp_celsius, &arr_cfg);
        let k_reference = arrhenius_rate_constant(20.0, &arr_cfg);
        let accel_factor = if k_reference > 0.0 {
            k_current / k_reference
        } else {
            1.0
        };

        total_equivalent_years += period_duration * accel_factor;

        weighted_temp_sum += curr.temp_celsius * period_duration;
        total_weight += period_duration;
    }

    let avg_temp = if total_weight > 0.0 {
        weighted_temp_sum / total_weight
    } else {
        current_temp_c
    };

    let k_ref_20 = arrhenius_rate_constant(20.0, &arr_cfg);
    let half_life_ref = if k_ref_20 > 0.0 {
        0.693 / k_ref_20
    } else {
        1.0e6
    };
    let half_life_years_ref = half_life_ref / (365.25 * 24.0 * 3600.0);

    let k_avg = arrhenius_rate_constant(avg_temp, &arr_cfg);
    let half_life_current = if k_avg > 0.0 {
        0.693 / k_avg
    } else {
        1.0e6
    };
    let half_life_years_current = half_life_current / (365.25 * 24.0 * 3600.0);

    let equivalent_time_s = total_equivalent_years * 365.25 * 24.0 * 3600.0;
    let decay = (-k_ref_20 * equivalent_time_s).exp();
    let remaining_pct = (initial_collagen_fraction * decay * 100.0).clamp(0.0, 100.0);

    let (equivalent_after_perturbation, remaining_after_perturbation, perturbation_applied, perturbation_loss_pct, pert_data) =
        if let Some(pert) = perturbation {
            let exposure_years = pert.exposure_duration_days / 365.25;
            let k_surface = arrhenius_rate_constant(pert.surface_temp_c, &arr_cfg);
            let accel_surface = if k_ref_20 > 0.0 {
                k_surface / k_ref_20
            } else {
                1.0
            };
            let perturbation_equivalent_years = exposure_years
                * accel_surface
                * pert.thermal_shock_amplification_factor.max(1.0);

            let total_eq_years = total_equivalent_years + perturbation_equivalent_years;
            let total_eq_s = total_eq_years * 365.25 * 24.0 * 3600.0;
            let decay_total = (-k_ref_20 * total_eq_s).exp();
            let remaining_after =
                (initial_collagen_fraction * decay_total * 100.0).clamp(0.0, 100.0);

            let loss_pct = if remaining_pct > 0.0 {
                ((remaining_pct - remaining_after) / remaining_pct * 100.0).max(0.0)
            } else {
                0.0
            };

            let data = (pert.surface_temp_c, pert.exposure_duration_days);
            (total_eq_years, remaining_after, true, loss_pct, Some(data))
        } else {
            (total_equivalent_years, remaining_pct, false, 0.0, None)
        };

    let cpi_raw = remaining_after_perturbation;
    let cpi_score = cpi_raw.clamp(0.0, 100.0);

    let grade = if cpi_score >= 85.0 {
        "A级 (极佳保存: 可开展古DNA/稳定同位素/氨基酸分析)".to_string()
    } else if cpi_score >= 65.0 {
        "B级 (良好保存: 可开展稳定同位素及常规结构分析)".to_string()
    } else if cpi_score >= 40.0 {
        "C级 (一般保存: 仅可开展元素组成与宏观结构分析)".to_string()
    } else if cpi_score >= 15.0 {
        "D级 (较差保存: 仅限形态学与矿化研究)".to_string()
    } else {
        "E级 (严重降解: 仅存矿化骨架, 有机质分析无效)".to_string()
    };

    let interpretation = if perturbation_applied {
        let (surf_temp, exp_days) = pert_data.unwrap();
        let note = if perturbation_loss_pct > 10.0 {
            format!("WARNING: Excavation thermal perturbation significant: {:.0}C surface for {:.1} days causes {:.1}% additional collagen loss.",
                surf_temp, exp_days, perturbation_loss_pct)
        } else {
            format!("OK: Excavation thermal perturbation manageable: {:.0}C surface for {:.1} days causes only {:.1}% loss.",
                surf_temp, exp_days, perturbation_loss_pct)
        };

        if cpi_score >= 85.0 {
            format!("{}Excellent preservation. Remaining collagen: {:.1}% (before excavation: {:.1}%, after: {:.1}%), half-life ~{:.1} years. Prioritize excavation, keep samples cold.",
                note, remaining_after_perturbation, remaining_pct, remaining_after_perturbation, half_life_years_current)
        } else if cpi_score >= 65.0 {
            format!("{}Good preservation. Remaining: {:.1}% (before {:.1}%, after {:.1}%), half-life ~{:.1} years. Excavate as planned, cold storage within 72h.",
                note, remaining_after_perturbation, remaining_pct, remaining_after_perturbation, half_life_years_current)
        } else if cpi_score >= 40.0 {
            format!("{}Fair preservation. Remaining: {:.1}% (before {:.1}%, after {:.1}%), half-life ~{:.1} years. Excavate soon, consider PEG embedding.",
                note, remaining_after_perturbation, remaining_pct, remaining_after_perturbation, half_life_years_current)
        } else if cpi_score >= 15.0 {
            format!("{}Poor preservation. Only {:.1}% remaining (before {:.1}%, after {:.1}%), half-life ~{:.1} years. Rescue excavation needed, cold chain transport.",
                note, remaining_after_perturbation, remaining_pct, remaining_after_perturbation, half_life_years_current)
        } else {
            format!("{}Severely degraded. Only {:.1}% left (before {:.1}%, after {:.1}%), half-life ~{:.1} years. Organic structure lost. Focus on morphological/mineral analysis.",
                note, remaining_after_perturbation, remaining_pct, remaining_after_perturbation, half_life_years_current)
        }
    } else {
        if cpi_score >= 85.0 {
            format!("Excellent preservation. Remaining collagen: {:.1}%, half-life ~{:.1} years. Prioritize excavation, store at -20C.",
                remaining_pct, half_life_years_current)
        } else if cpi_score >= 65.0 {
            format!("Good preservation. Remaining: {:.1}%, half-life ~{:.1} years. Excavate as planned, cold storage within 72h.",
                remaining_pct, half_life_years_current)
        } else if cpi_score >= 40.0 {
            format!("Fair preservation. Remaining: {:.1}%, half-life ~{:.1} years. Excavate within 1-2 years, consider PEG embedding.",
                remaining_pct, half_life_years_current)
        } else if cpi_score >= 15.0 {
            format!("Poor preservation. Only {:.1}% left, half-life ~{:.1} years. Rescue excavation, on-site consolidation recommended.",
                remaining_pct, half_life_years_current)
        } else {
            format!("Severely degraded. Only {:.1}% left, organic structure lost. Focus on morphological analysis.",
                remaining_pct)
        }
    };

    CollagenPreservationIndex {
        cpi_score,
        cpi_grade: grade,
        equivalent_years_at_20c: total_equivalent_years,
        equivalent_years_after_perturbation_at_20c: equivalent_after_perturbation,
        remaining_collagen_pct: remaining_pct,
        remaining_after_perturbation_pct: remaining_after_perturbation,
        predicted_half_life_years: half_life_years_current,
        initial_half_life_years: half_life_years_ref,
        temperature_history: history,
        activation_energy,
        burial_years,
        average_temp_c: avg_temp,
        excavation_perturbation_applied: perturbation_applied,
        perturbation_loss_pct,
        interpretation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_ea_has_longer_half_life() {
        let cpi_low_ea = calculate_cpi(60_000.0, 1000.0, 15.0, None, 1.0, None);
        let cpi_high_ea = calculate_cpi(120_000.0, 1000.0, 15.0, None, 1.0, None);
        assert!(cpi_high_ea.predicted_half_life_years > cpi_low_ea.predicted_half_life_years * 2.0);
    }

    #[test]
    fn test_low_temp_preserves_better() {
        let cpi_cold = calculate_cpi(85_000.0, 1000.0, 4.0, None, 1.0, None);
        let cpi_warm = calculate_cpi(85_000.0, 1000.0, 25.0, None, 1.0, None);
        assert!(cpi_cold.cpi_score > cpi_warm.cpi_score);
    }

    #[test]
    fn test_short_burial_high_preservation() {
        let cpi = calculate_cpi(85_000.0, 10.0, 15.0, None, 1.0, None);
        assert!(cpi.cpi_score > 80.0);
    }

    #[test]
    fn test_long_burial_low_preservation() {
        let cpi = calculate_cpi(85_000.0, 50_000.0, 20.0, None, 1.0, None);
        assert!(cpi.cpi_score < 10.0);
    }

    #[test]
    fn test_excavation_perturbation_reduces_cpi() {
        let no_perturb = calculate_cpi(85_000.0, 500.0, 15.0, None, 1.0, None);
        let with_perturb = calculate_cpi(
            85_000.0,
            500.0,
            15.0,
            None,
            1.0,
            Some(ExcavationThermalPerturbation {
                surface_temp_c: 40.0,
                exposure_duration_days: 3.0,
                thermal_shock_amplification_factor: 3.0,
            }),
        );
        assert!(with_perturb.cpi_score < no_perturb.cpi_score);
        assert!(with_perturb.perturbation_loss_pct > 0.0);
    }

    #[test]
    fn test_cpi_score_bounds_0_100() {
        let cpi = calculate_cpi(50_000.0, 1_000_000.0, 30.0, None, 1.0, None);
        assert!(cpi.cpi_score >= 0.0 && cpi.cpi_score <= 100.0);
    }

    #[test]
    fn test_interpretation_not_empty() {
        let cpi = calculate_cpi(85_000.0, 500.0, 15.0, None, 1.0, None);
        assert!(!cpi.interpretation.is_empty());
    }

    #[test]
    fn test_equivalent_time_method_valid() {
        let history = vec![
            TemperatureHistoryPoint { years_bp: 100.0, temp_celsius: 20.0 },
            TemperatureHistoryPoint { years_bp: 0.0, temp_celsius: 20.0 },
        ];
        let cpi = calculate_cpi(85_000.0, 100.0, 20.0, Some(history), 1.0, None);
        assert!((cpi.equivalent_years_at_20c - 100.0).abs() < 1.0);
    }
}
