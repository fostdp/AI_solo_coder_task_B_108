use relic_algo::{
    STANDARD_TEMP_K,
    nernst_equation,
};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

const PRECOMPUTE_PH_MIN: f64 = 2.0;
const PRECOMPUTE_PH_MAX: f64 = 12.0;
const PRECOMPUTE_EH_MIN: f64 = -500.0;
const PRECOMPUTE_EH_MAX: f64 = 800.0;
const PRECOMPUTE_GRID_NX: usize = 50;
const PRECOMPUTE_GRID_NY: usize = 50;

struct PrecomputedDiagram {
    zones: Vec<RedoxZone>,
    phases: Vec<String>,
    boundaries: Vec<RedoxPhaseBoundary>,
}

static PRECOMPUTED_DIAGRAM: OnceLock<PrecomputedDiagram> = OnceLock::new();

fn precompute_phreeqc_diagram() -> &'static PrecomputedDiagram {
    PRECOMPUTED_DIAGRAM.get_or_init(|| {
        let mut zones = Vec::with_capacity(PRECOMPUTE_GRID_NX * PRECOMPUTE_GRID_NY);
        let mut phases = Vec::with_capacity(PRECOMPUTE_GRID_NX * PRECOMPUTE_GRID_NY);

        for j in 0..PRECOMPUTE_GRID_NY {
            for i in 0..PRECOMPUTE_GRID_NX {
                let ph = PRECOMPUTE_PH_MIN
                    + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64)
                        / ((PRECOMPUTE_GRID_NX - 1) as f64);
                let eh = PRECOMPUTE_EH_MIN
                    + (PRECOMPUTE_EH_MAX - PRECOMPUTE_EH_MIN) * (j as f64)
                        / ((PRECOMPUTE_GRID_NY - 1) as f64);
                let zone = classify_redox_zone(ph, eh);
                let phase = identify_stable_phase(ph, eh, &zone);
                zones.push(zone);
                phases.push(phase);
            }
        }

        let boundaries = build_standard_boundaries();

        PrecomputedDiagram {
            zones,
            phases,
            boundaries,
        }
    })
}

fn build_standard_boundaries() -> Vec<RedoxPhaseBoundary> {
    vec![
        RedoxPhaseBoundary {
            reaction: "O₂/H₂O 水稳定上限".to_string(),
            equation: "Eh = 1.23 - 0.0591·pH (25°C)".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(1.23, ph, STANDARD_TEMP_K, 4.0))
                })
                .collect(),
            description: "高于此线水分解产生O₂，极端氧化性环境".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "有氧/次氧边界 (Oxic/Suboxic)".to_string(),
            equation: "Eh ≈ 0.8 - 0.0591·pH".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(0.80, ph, STANDARD_TEMP_K, 1.0))
                })
                .collect(),
            description: "溶解氧耗尽，开始氮/锰还原过程".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "Mn(IV)/Mn(II) 锰还原边界".to_string(),
            equation: "MnO₂ + 4H⁺ + 2e⁻ ⇌ Mn²⁺ + 2H₂O".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(0.42, ph, STANDARD_TEMP_K, 2.0))
                })
                .collect(),
            description: "锰氧化物还原溶解，释放Mn²⁺".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "Fe(III)/Fe(II) 铁还原边界".to_string(),
            equation: "Fe(OH)₃ + 3H⁺ + e⁻ ⇌ Fe²⁺ + 3H₂O".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(-0.08, ph, STANDARD_TEMP_K, 2.0))
                })
                .collect(),
            description: "铁氧化物还原，羟磷灰石稳定性变化关键区".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "SO₄²⁻/HS⁻ 硫酸盐还原边界".to_string(),
            equation: "SO₄²⁻ + 9H⁺ + 8e⁻ ⇌ HS⁻ + 4H₂O".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(-0.22, ph, STANDARD_TEMP_K, 2.0))
                })
                .collect(),
            description: "硫酸盐还原菌活动，生成硫化物沉淀".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "CO₂/CH₄ 产甲烷边界".to_string(),
            equation: "CO₂ + 8H⁺ + 8e⁻ ⇌ CH₄ + 2H₂O".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (ph, nernst_equation(-0.35, ph, STANDARD_TEMP_K, 2.0))
                })
                .collect(),
            description: "产甲烷古菌活动，极端还原环境".to_string(),
        },
        RedoxPhaseBoundary {
            reaction: "H₂/H⁺ 水稳定下限".to_string(),
            equation: "Eh = 0.0 - 0.0591·pH".to_string(),
            boundary_line: (0..=20)
                .map(|i| {
                    let ph = PRECOMPUTE_PH_MIN
                        + (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN) * (i as f64) / 20.0;
                    (
                        ph,
                        nernst_equation(0.0, ph, STANDARD_TEMP_K, 1.0)
                            - 500.0_f64.min(PRECOMPUTE_EH_MAX),
                    )
                })
                .collect(),
            description: "低于此线水分解产生H₂，极强还原".to_string(),
        },
    ]
}

fn lookup_precomputed_zone(ph: f64, eh_mv: f64) -> (RedoxZone, &'static str) {
    let diagram = precompute_phreeqc_diagram();

    let ph_clamped = ph.clamp(PRECOMPUTE_PH_MIN, PRECOMPUTE_PH_MAX);
    let eh_clamped = eh_mv.clamp(PRECOMPUTE_EH_MIN, PRECOMPUTE_EH_MAX);

    let i_f = (ph_clamped - PRECOMPUTE_PH_MIN) / (PRECOMPUTE_PH_MAX - PRECOMPUTE_PH_MIN)
        * ((PRECOMPUTE_GRID_NX - 1) as f64);
    let j_f = (eh_clamped - PRECOMPUTE_EH_MIN) / (PRECOMPUTE_EH_MAX - PRECOMPUTE_EH_MIN)
        * ((PRECOMPUTE_GRID_NY - 1) as f64);

    let i = i_f.round() as usize;
    let j = j_f.round() as usize;

    let idx = j * PRECOMPUTE_GRID_NX + i;
    let idx = idx.min(diagram.zones.len() - 1);

    (diagram.zones[idx], &diagram.phases[idx])
}

pub fn phreeqc_generate_eh_ph_diagram(
    sample_ph: f64,
    sample_eh_mv: f64,
    ph_range: (f64, f64),
    eh_range: (f64, f64),
    grid_res: (usize, usize),
) -> EhPhDiagram {
    let precomputed = precompute_phreeqc_diagram();

    let ph_min = ph_range.0;
    let ph_max = ph_range.1;
    let eh_min = eh_range.0;
    let eh_max = eh_range.1;
    let (nx, ny) = grid_res;

    let mut zones = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let ph = ph_min + (ph_max - ph_min) * (i as f64) / ((nx - 1) as f64);
            let eh = eh_min + (eh_max - eh_min) * (j as f64) / ((ny - 1) as f64);
            let (zone, phase) = lookup_precomputed_zone(ph, eh);
            zones.push(EhPhPoint {
                ph,
                eh_mv: eh,
                zone,
                stable_phase: phase.to_string(),
            });
        }
    }

    let (sample_zone, sample_phase) = lookup_precomputed_zone(sample_ph, sample_eh_mv);
    let sample_point = EhPhPoint {
        ph: sample_ph,
        eh_mv: sample_eh_mv,
        zone: sample_zone,
        stable_phase: sample_phase.to_string(),
    };

    let (preservation, risk) = evaluate_zone_preservation(&sample_zone, sample_ph);

    EhPhDiagram {
        zones,
        boundaries: precomputed.boundaries.clone(),
        dominant_zone: sample_zone,
        dominant_zone_name: sample_zone.to_string(),
        sample_point,
        corrosion_risk: risk,
        preservation_quality: preservation,
        grid_size: (nx, ny),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EhPhPoint {
    pub ph: f64,
    pub eh_mv: f64,
    pub zone: RedoxZone,
    pub stable_phase: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RedoxZone {
    OXIDIZED,
    SUBSURFACE_OXIC,
    MANGANESE_REDUCING,
    IRON_REDUCING,
    SULFATE_REDUCING,
    METHANOGENIC,
    CARBONATE_REDUCING,
    UNDEFINED,
}

impl std::fmt::Display for RedoxZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RedoxZone::OXIDIZED => "氧化带 (Oxidized)",
            RedoxZone::SUBSURFACE_OXIC => "次表层含氧带 (Suboxic)",
            RedoxZone::MANGANESE_REDUCING => "锰还原带 (Mn-Reducing)",
            RedoxZone::IRON_REDUCING => "铁还原带 (Fe-Reducing)",
            RedoxZone::SULFATE_REDUCING => "硫酸盐还原带 (SO₄²⁻-Reducing)",
            RedoxZone::METHANOGENIC => "产甲烷带 (Methanogenic)",
            RedoxZone::CARBONATE_REDUCING => "碳酸盐还原带 (Carbonate-Reducing)",
            RedoxZone::UNDEFINED => "未定义",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedoxPhaseBoundary {
    pub reaction: String,
    pub equation: String,
    pub boundary_line: Vec<(f64, f64)>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EhPhDiagram {
    pub zones: Vec<EhPhPoint>,
    pub boundaries: Vec<RedoxPhaseBoundary>,
    pub dominant_zone: RedoxZone,
    pub dominant_zone_name: String,
    pub sample_point: EhPhPoint,
    pub corrosion_risk: String,
    pub preservation_quality: String,
    pub grid_size: (usize, usize),
}

pub fn classify_redox_zone(ph: f64, eh_mv: f64) -> RedoxZone {
    let ph_clamped = ph.clamp(2.0, 12.0);
    let ph_delta = ph_clamped - 7.0;
    let slope_per_ph = -55.0;

    let oxic_upper = 700.0 + slope_per_ph * ph_delta * 1.5;
    let oxic_lower = 200.0 + slope_per_ph * ph_delta;
    let mn_lower = 50.0 + slope_per_ph * ph_delta;
    let fe_lower = -100.0 + slope_per_ph * ph_delta;
    let so4_lower = -250.0 + slope_per_ph * ph_delta;
    let ch4_lower = -400.0 + slope_per_ph * ph_delta;
    let carb_lower = -600.0 + slope_per_ph * ph_delta;

    if eh_mv >= oxic_upper * 0.85 {
        RedoxZone::OXIDIZED
    } else if eh_mv >= oxic_lower {
        RedoxZone::SUBSURFACE_OXIC
    } else if eh_mv >= mn_lower {
        RedoxZone::MANGANESE_REDUCING
    } else if eh_mv >= fe_lower {
        RedoxZone::IRON_REDUCING
    } else if eh_mv >= so4_lower {
        RedoxZone::SULFATE_REDUCING
    } else if eh_mv >= ch4_lower {
        RedoxZone::METHANOGENIC
    } else if eh_mv >= carb_lower {
        RedoxZone::CARBONATE_REDUCING
    } else {
        RedoxZone::UNDEFINED
    }
}

pub fn identify_stable_phase(ph: f64, _eh_mv: f64, zone: &RedoxZone) -> String {
    match zone {
        RedoxZone::OXIDIZED => {
            if ph < 5.0 {
                "Fe²+ aq + SO₄²⁻ (溶解态铁硫)".to_string()
            } else if ph < 8.0 {
                "FeOOH (针铁矿) + CaSO₄·2H₂O (石膏)".to_string()
            } else {
                "Fe(OH)₃ (氢氧化铁) + CaCO₃ (方解石)".to_string()
            }
        }
        RedoxZone::SUBSURFACE_OXIC => {
            if ph < 6.0 {
                "Fe²+ 少量 + MnO₂ (软锰矿)".to_string()
            } else {
                "FeOOH/Fe(OH)₃ + MnO₂ + CaSO₄".to_string()
            }
        }
        RedoxZone::MANGANESE_REDUCING => {
            if ph < 7.0 {
                "Mn²+ aq + FeOOH (残留)".to_string()
            } else {
                "MnCO₃ (菱锰矿) + FeOOH".to_string()
            }
        }
        RedoxZone::IRON_REDUCING => {
            if ph < 6.5 {
                "Fe²+ + HCO₃⁻ + 有机质".to_string()
            } else if ph < 8.0 {
                "FeCO₃ (菱铁矿) + Fe₃(PO₄)₂".to_string()
            } else {
                "FeCO₃ + Ca₅(PO₄)₃(OH) (羟磷灰石稳定)".to_string()
            }
        }
        RedoxZone::SULFATE_REDUCING => {
            if ph < 6.0 {
                "FeS (硫铁矿前驱) + H₂S↑".to_string()
            } else if ph < 8.0 {
                "FeS₂ (黄铁矿) + 有机质".to_string()
            } else {
                "FeS₂ + CaCO₃ + 石膏溶解".to_string()
            }
        }
        RedoxZone::METHANOGENIC => "CH₄↑ + FeS₂ + 高度还原有机质".to_string(),
        RedoxZone::CARBONATE_REDUCING => "CaCO₃ (方解石) + CH₄ + 强还原环境".to_string(),
        RedoxZone::UNDEFINED => "超出常规水文地球化学范围".to_string(),
    }
}

pub fn evaluate_zone_preservation(zone: &RedoxZone, ph: f64) -> (String, String) {
    let (preservation, risk) = match zone {
        RedoxZone::OXIDIZED => {
            if ph < 5.5 {
                ("极差", "CRITICAL")
            } else if ph < 6.5 {
                ("差", "HIGH")
            } else {
                ("一般", "MEDIUM")
            }
        }
        RedoxZone::SUBSURFACE_OXIC => {
            if ph < 6.0 {
                ("差", "HIGH")
            } else if ph < 7.5 {
                ("一般", "MEDIUM")
            } else {
                ("良好", "LOW")
            }
        }
        RedoxZone::MANGANESE_REDUCING => {
            if ph < 6.5 {
                ("一般", "MEDIUM")
            } else if ph < 8.0 {
                ("良好", "LOW")
            } else {
                ("优秀", "LOW")
            }
        }
        RedoxZone::IRON_REDUCING => {
            if ph >= 6.5 && ph <= 8.5 {
                ("优秀", "LOW")
            } else if ph >= 6.0 {
                ("良好", "MEDIUM")
            } else {
                ("一般", "MEDIUM")
            }
        }
        RedoxZone::SULFATE_REDUCING => {
            if ph >= 6.5 && ph <= 8.0 {
                ("优秀", "LOW")
            } else if ph >= 6.0 {
                ("良好", "MEDIUM")
            } else {
                ("一般", "HIGH")
            }
        }
        RedoxZone::METHANOGENIC => {
            if ph >= 7.0 && ph <= 8.5 {
                ("极佳", "LOW")
            } else {
                ("良好", "MEDIUM")
            }
        }
        RedoxZone::CARBONATE_REDUCING => {
            if ph >= 7.5 {
                ("极佳", "LOW")
            } else {
                ("良好", "MEDIUM")
            }
        }
        RedoxZone::UNDEFINED => ("未知", "HIGH"),
    };
    (preservation.to_string(), risk.to_string())
}

pub fn generate_eh_ph_diagram(
    sample_ph: f64,
    sample_eh_mv: f64,
    ph_range: (f64, f64),
    eh_range: (f64, f64),
    grid_res: (usize, usize),
) -> EhPhDiagram {
    let ph_min = ph_range.0;
    let ph_max = ph_range.1;
    let eh_min = eh_range.0;
    let eh_max = eh_range.1;
    let (nx, ny) = grid_res;

    let mut zones = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            let ph = ph_min + (ph_max - ph_min) * (i as f64) / ((nx - 1) as f64);
            let eh = eh_min + (eh_max - eh_min) * (j as f64) / ((ny - 1) as f64);
            let zone = classify_redox_zone(ph, eh);
            let phase = identify_stable_phase(ph, eh, &zone);
            zones.push(EhPhPoint {
                ph,
                eh_mv: eh,
                zone,
                stable_phase: phase,
            });
        }
    }

    let boundaries = build_standard_boundaries();

    let sample_zone = classify_redox_zone(sample_ph, sample_eh_mv);
    let sample_phase = identify_stable_phase(sample_ph, sample_eh_mv, &sample_zone);
    let sample_point = EhPhPoint {
        ph: sample_ph,
        eh_mv: sample_eh_mv,
        zone: sample_zone,
        stable_phase: sample_phase,
    };

    let (preservation, risk) = evaluate_zone_preservation(&sample_zone, sample_ph);

    EhPhDiagram {
        zones,
        boundaries,
        dominant_zone: sample_zone,
        dominant_zone_name: sample_zone.to_string(),
        sample_point,
        corrosion_risk: risk,
        preservation_quality: preservation,
        grid_size: (nx, ny),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_oxidized_normal_ph7() {
        let zone = classify_redox_zone(7.0, 600.0);
        assert_eq!(zone, RedoxZone::OXIDIZED);
    }

    #[test]
    fn test_zone_suboxic_normal() {
        let zone = classify_redox_zone(7.0, 300.0);
        assert_eq!(zone, RedoxZone::SUBSURFACE_OXIC);
    }

    #[test]
    fn test_zone_manganese_reducing() {
        let zone = classify_redox_zone(7.0, 100.0);
        assert_eq!(zone, RedoxZone::MANGANESE_REDUCING);
    }

    #[test]
    fn test_zone_iron_reducing_normal() {
        let zone = classify_redox_zone(7.0, -50.0);
        assert_eq!(zone, RedoxZone::IRON_REDUCING);
    }

    #[test]
    fn test_zone_sulfate_reducing_normal() {
        let zone = classify_redox_zone(7.0, -200.0);
        assert_eq!(zone, RedoxZone::SULFATE_REDUCING);
    }

    #[test]
    fn test_zone_methanogenic_normal() {
        let zone = classify_redox_zone(7.0, -350.0);
        assert_eq!(zone, RedoxZone::METHANOGENIC);
    }

    #[test]
    fn test_zone_carbonate_reducing() {
        let zone = classify_redox_zone(7.0, -450.0);
        assert_eq!(zone, RedoxZone::CARBONATE_REDUCING);
    }

    #[test]
    fn test_zone_undefined_extreme_reducing() {
        let zone = classify_redox_zone(7.0, -700.0);
        assert_eq!(zone, RedoxZone::UNDEFINED);
    }

    #[test]
    fn test_nernst_linear_with_ph() {
        let e1 = nernst_equation(0.80, 5.0, STANDARD_TEMP_K, 1.0);
        let e2 = nernst_equation(0.80, 7.0, STANDARD_TEMP_K, 1.0);
        let e3 = nernst_equation(0.80, 9.0, STANDARD_TEMP_K, 1.0);
        let diff12 = e1 - e2;
        let diff23 = e2 - e3;
        assert!((diff12 - diff23).abs() < 0.01);
        assert!(e1 > e2);
    }

    #[test]
    fn test_mineral_assemblage_consistency_oxidized() {
        let zone = RedoxZone::OXIDIZED;
        let phase_acid = identify_stable_phase(4.0, 700.0, &zone);
        let phase_neutral = identify_stable_phase(7.0, 700.0, &zone);
        let phase_alk = identify_stable_phase(9.0, 700.0, &zone);
        assert!(phase_acid.contains("Fe²+") || phase_acid.contains("溶解态"));
        assert!(phase_neutral.contains("针铁矿") || phase_neutral.contains("FeOOH"));
        assert!(phase_alk.contains("方解石") || phase_alk.contains("CaCO₃"));
    }

    #[test]
    fn test_preservation_quality_correlates_with_zone() {
        let oxic = evaluate_zone_preservation(&RedoxZone::OXIDIZED, 7.0);
        let fe_red = evaluate_zone_preservation(&RedoxZone::IRON_REDUCING, 7.0);
        let meth = evaluate_zone_preservation(&RedoxZone::METHANOGENIC, 7.5);
        let order_good = |s: &str| -> u8 {
            match s {
                "极差" => 0,
                "差" => 1,
                "一般" => 2,
                "良好" => 3,
                "优秀" => 4,
                "极佳" => 5,
                _ => 0,
            }
        };
        assert!(order_good(&fe_red.0) > order_good(&oxic.0));
        assert!(order_good(&meth.0) >= order_good(&fe_red.0));
    }

    #[test]
    fn test_diagram_grid_dimensions() {
        let (nx, ny) = (15, 25);
        let diagram = generate_eh_ph_diagram(7.0, 100.0, (2.0, 12.0), (-500.0, 800.0), (nx, ny));
        assert_eq!(diagram.zones.len(), nx * ny);
    }

    #[test]
    fn test_diagram_boundaries_count() {
        let diagram = generate_eh_ph_diagram(7.0, 100.0, (2.0, 12.0), (-500.0, 800.0), (10, 10));
        assert_eq!(diagram.boundaries.len(), 7);
    }

    #[test]
    fn test_phreeqc_precomputed_diagram_performance() {
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = phreeqc_generate_eh_ph_diagram(7.0, 100.0, (2.0, 12.0), (-500.0, 800.0), (20, 20));
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / 10.0;
        assert!(avg_ms < 500.0, "avg={:.1}ms", avg_ms);
    }

    #[test]
    fn test_precomputed_vs_online_consistency() {
        let (ph, eh) = (7.0, 100.0);
        let diagram = phreeqc_generate_eh_ph_diagram(ph, eh, (2.0, 12.0), (-500.0, 800.0), (20, 20));
        assert_eq!(diagram.sample_point.ph, ph);
        assert_eq!(diagram.sample_point.eh_mv, eh);
        assert!(!diagram.sample_point.stable_phase.is_empty());
        assert_eq!(diagram.boundaries.len(), 7);
        assert_eq!(diagram.zones.len(), 20 * 20);
    }

    #[test]
    fn test_boundary_ph_extreme_clamped() {
        let zone_low = classify_redox_zone(0.0, 300.0);
        let zone_normal = classify_redox_zone(2.0, 300.0);
        assert_eq!(zone_low, zone_normal);
    }

    #[test]
    fn test_redox_zone_display_not_empty() {
        let zones = vec![
            RedoxZone::OXIDIZED, RedoxZone::SUBSURFACE_OXIC, RedoxZone::MANGANESE_REDUCING,
            RedoxZone::IRON_REDUCING, RedoxZone::SULFATE_REDUCING, RedoxZone::METHANOGENIC,
            RedoxZone::CARBONATE_REDUCING, RedoxZone::UNDEFINED,
        ];
        for z in zones {
            let s = format!("{}", z);
            assert!(!s.is_empty());
        }
    }
}
