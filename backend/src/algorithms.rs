use crate::models::CorrosionAnalysis;
use crate::kinetics;
use chrono::Utc;

pub use relic_algo::{
    ArrheniusConfig, CalciumPhosphateConfig, MassBalanceState,
    GAS_CONSTANT_R, ACTIVATION_ENERGY_EA, PRE_EXPONENTIAL_A, REFERENCE_TEMP_K,
    HYDROXYAPATITE_CA_P_STOICHIOMETRIC,
    FARADAY_F, MOLAR_GAS_R, STANDARD_TEMP_K, ELECTRONS_TRANSFERRED,
    arrhenius_rate_constant, ph_catalysis_factor, orp_effect_factor,
    collagen_hydrolysis_rate, collagen_degradation_percent,
    expected_collagen_deg_elapsed_months, hap_solubility, dissolution_rate,
    ca_p_ratio_current, ca_p_ratio_predicted, ca_to_ppm,
    corrosion_rate_um_per_year, estimate_corrosion_depth_um,
    assess_risk_level, estimate_microbial_biomass, nernst_equation,
};

pub fn perform_full_analysis(
    relic_id: &str,
    grid_x: f64,
    grid_y: f64,
    ph: f64,
    temperature: f64,
    ca_ppm: f64,
    orp_mv: f64,
    elapsed_months: f64,
) -> CorrosionAnalysis {
    let arr_cfg = ArrheniusConfig::default();
    let mm_cfg = kinetics::MichaelisMentenConfig::default();

    let k_abiotic = collagen_hydrolysis_rate(temperature, ph, orp_mv, Some(&arr_cfg));

    let microbial_biomass = estimate_microbial_biomass(temperature, ph, orp_mv);
    let substrate_conc = 1.0;
    let k_enzyme = kinetics::enzyme_hydrolysis_rate(
        substrate_conc, microbial_biomass, temperature, ph, &mm_cfg,
    );

    let coll_rate = k_abiotic + k_enzyme;
    let coll_deg_pct = expected_collagen_deg_elapsed_months(coll_rate, elapsed_months);

    let enzyme_contribution_pct = if coll_rate > 0.0 {
        (k_enzyme / coll_rate) * 100.0
    } else {
        0.0
    };

    let elapsed_days = elapsed_months * 30.0;
    let ca_pred = ca_p_ratio_predicted(ph, temperature, elapsed_days, None);
    let diss_rate = dissolution_rate(ph, temperature, ca_ppm, 0.5, &CalciumPhosphateConfig::default());
    let cor_rate = corrosion_rate_um_per_year(coll_rate, diss_rate, ph);
    let cor_depth = estimate_corrosion_depth_um(cor_rate, elapsed_days);
    let risk = assess_risk_level(ph, ca_ppm, coll_deg_pct, cor_depth);

    CorrosionAnalysis {
        relic_id: relic_id.to_string(),
        grid_x,
        grid_y,
        ph,
        temperature,
        ca_concentration: ca_ppm,
        orp: orp_mv,
        collagen_deg_rate: coll_rate,
        collagen_deg_percent: coll_deg_pct,
        abiotic_rate: k_abiotic,
        enzyme_rate: k_enzyme,
        enzyme_contribution_pct,
        microbial_biomass,
        ca_p_ratio: if ca_ppm > 0.0 {
            let ca_cfg = CalciumPhosphateConfig::default();
            let ca_mg = ca_ppm * 0.5;
            let p_mg = ca_mg / HYDROXYAPATITE_CA_P_STOICHIOMETRIC * (1.0 - (coll_deg_pct / 100.0) * 0.3);
            if p_mg > 1e-9 {
                ca_mg / p_mg
            } else {
                HYDROXYAPATITE_CA_P_STOICHIOMETRIC
            }
        } else {
            HYDROXYAPATITE_CA_P_STOICHIOMETRIC
        },
        ca_p_ratio_predicted: ca_pred,
        corrosion_rate: cor_rate,
        corrosion_depth_um: cor_depth,
        risk_level: risk,
        timestamp: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrhenius_normal_temp() {
        let k = arrhenius_rate_constant(25.0, &ArrheniusConfig::default());
        assert!(k > 0.0);
    }

    #[test]
    fn test_ph_factor_acidic() {
        let cfg = ArrheniusConfig::default();
        let f_ac = ph_catalysis_factor(4.0, &cfg);
        let f_ne = ph_catalysis_factor(7.0, &cfg);
        assert!(f_ac > f_ne);
    }

    #[test]
    fn test_collagen_deg_increases() {
        let r = collagen_hydrolysis_rate(25.0, 7.0, 100.0, None);
        let p1 = expected_collagen_deg_elapsed_months(r, 1.0);
        let p12 = expected_collagen_deg_elapsed_months(r, 12.0);
        assert!(p12 > p1);
    }

    #[test]
    fn test_ca_p_ratio_predicted() {
        let ratio = ca_p_ratio_predicted(7.0, 20.0, 365.0, None);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_risk_level_ph_low() {
        let r = assess_risk_level(4.0, 50.0, 5.0, 20.0);
        assert_eq!(r, "HIGH");
    }

    #[test]
    fn test_full_analysis() {
        let a = perform_full_analysis("RLC-0001", 5.0, 5.0, 6.5, 18.0, 80.0, 200.0, 3.0);
        assert!(a.collagen_deg_percent >= 0.0 && a.collagen_deg_percent <= 100.0);
        assert!(a.corrosion_depth_um >= 0.0);
    }
}
