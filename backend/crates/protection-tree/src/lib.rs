use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};

const NUM_RF_TREES: usize = 5;

const TRAINING_DATA: &[([f64; 2], usize)] = &[
    ([3.0, 30.0], 0), ([3.5, 20.0], 0), ([4.0, 40.0], 0), ([4.2, 60.0], 0),
    ([4.4, 80.0], 0), ([4.8, 50.0], 0), ([5.0, 30.0], 0), ([5.2, 80.0], 1),
    ([5.5, 10.0], 1), ([5.8, 40.0], 1), ([6.0, 20.0], 2), ([6.2, 60.0], 2),
    ([6.3, 30.0], 2), ([6.5, 80.0], 2), ([6.8, 50.0], 2), ([7.0, 30.0], 3),
    ([7.0, 80.0], 3), ([7.2, 100.0], 3), ([7.5, 60.0], 3), ([7.8, 150.0], 4),
    ([8.0, 250.0], 4), ([8.2, 400.0], 4), ([8.5, 300.0], 5), ([9.0, 200.0], 5),
    ([9.5, 150.0], 5), ([10.0, 100.0], 5), ([5.0, 10.0], 1), ([5.5, 5.0], 1),
    ([6.0, 80.0], 2), ([6.5, 40.0], 2), ([7.0, 50.0], 3), ([7.5, 200.0], 4),
    ([8.0, 350.0], 4), ([4.5, 50.0], 0), ([5.0, 70.0], 1),
];

const CLASS_LABELS: &[&str] = &[
    "PEG400-PBS_Buffered",
    "PEG200_Saturated_Calcium_Hydroxide",
    "PEG200_Pure",
    "DI_Water",
    "Ethanol_Water_30pct",
    "PEG200_Weak_Acid_Buffered",
];

const CLASS_LABELS_ZH: &[&str] = &[
    "pH缓冲的聚乙二醇400溶液",
    "PEG200饱和氢氧化钙溶液",
    "纯PEG200溶液",
    "去离子水 (Milli-Q级)",
    "30%乙醇-水溶液",
    "弱酸缓冲的PEG200溶液",
];

const CLASS_CONCENTRATIONS: &[f64] = &[30.0, 40.0, 50.0, 100.0, 30.0, 35.0];
const CLASS_EFFECTIVENESS: &[f64] = &[85.0, 92.0, 80.0, 78.0, 75.0, 70.0];

fn build_dataset() -> DatasetBase<Array2<f64>, Array1<usize>> {
    let n = TRAINING_DATA.len();
    let mut features = Vec::with_capacity(n * 2);
    let mut labels = Vec::with_capacity(n);

    for (feat, label) in TRAINING_DATA {
        features.push(feat[0]);
        features.push(feat[1]);
        labels.push(*label);
    }

    let x = Array2::from_shape_vec((n, 2), features).unwrap();
    let y = Array1::from_vec(labels);
    Dataset::new(x, y)
}

fn train_single_tree(_rng_seed: u64) -> DecisionTree<f64, usize> {
    let dataset = build_dataset();
    linfa_trees::DecisionTree::params()
        .max_depth(Some(5))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&dataset)
        .expect("Decision tree training failed")
}

fn train_forest() -> Vec<DecisionTree<f64, usize>> {
    (0..NUM_RF_TREES).map(|i| train_single_tree(i as u64)).collect()
}

fn predict_with_forest(forest: &[DecisionTree<f64, usize>], ph: f64, ca_ppm: f64) -> (usize, HashMap<usize, usize>) {
    let x = Array2::from_shape_vec((1, 2), vec![ph, ca_ppm]).unwrap();
    let mut votes: HashMap<usize, usize> = HashMap::new();

    for tree in forest {
        let pred = tree.predict(&x);
        let class_idx = pred[0] as usize;
        *votes.entry(class_idx).or_insert(0) += 1;
    }

    let best_class = votes
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(&class, _)| class)
        .unwrap_or(3);

    (best_class, votes)
}

fn classify_ph_condition(ph: f64) -> &'static str {
    if ph < 4.5 {
        "EXTREMELY_ACIDIC"
    } else if ph < 5.5 {
        "HIGHLY_ACIDIC"
    } else if ph < 6.5 {
        "MODERATELY_ACIDIC"
    } else if ph <= 7.5 {
        "NEUTRAL"
    } else if ph <= 8.5 {
        "MODERATELY_ALKALINE"
    } else if ph <= 9.5 {
        "HIGHLY_ALKALINE"
    } else {
        "EXTREMELY_ALKALINE"
    }
}

fn classify_ca_condition(ca_ppm: f64) -> &'static str {
    if ca_ppm < 30.0 {
        "VERY_LOW_CA"
    } else if ca_ppm < 80.0 {
        "LOW_CA"
    } else if ca_ppm < 200.0 {
        "NORMAL_CA"
    } else if ca_ppm < 400.0 {
        "HIGH_CA"
    } else {
        "VERY_HIGH_CA"
    }
}

fn classify_origination(orp_mv: f64) -> &'static str {
    if orp_mv > 250.0 {
        "HIGHLY_OXIDIZING"
    } else if orp_mv > 0.0 {
        "MODERATELY_OXIDIZING"
    } else if orp_mv > -150.0 {
        "MODERATELY_REDUCING"
    } else {
        "STRONGLY_REDUCING"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionRecommendation {
    pub primary_moisturizer: String,
    pub primary_moisturizer_zh: String,
    pub concentration_pct: f64,
    pub application_method: String,
    pub secondary_recommendations: Vec<String>,
    pub ph_neutralization_required: bool,
    pub neutralization_agent: Option<String>,
    pub expected_effectiveness_score: f64,
    pub estimated_stabilization_hours: f64,
    pub warnings: Vec<String>,
    pub decision_path: Vec<String>,
    pub materials_needed: Vec<ProtectionMaterial>,
    pub step_by_step_protocol: Vec<String>,
    pub prediction_confidence: f64,
    pub top_candidates: Vec<String>,
    pub top_candidates_zh: Vec<String>,
    pub candidate_confidences: Vec<f64>,
    pub data_missing_flags: Vec<String>,
    pub tree_vote_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionMaterial {
    pub name: String,
    pub name_zh: String,
    pub quantity_estimate: String,
    pub purpose: String,
    pub priority: String,
}

pub fn recommend_temporary_protection(
    ph: f64,
    ca_ppm: f64,
    orp_mv: f64,
    ambient_temp_c: f64,
    ambient_rh_pct: f64,
    burial_depth_m: f64,
    relic_category: &str,
) -> ProtectionRecommendation {
    let mut data_missing: Vec<String> = Vec::new();

    let ph = if ph.is_nan() || ph <= 0.0 {
        data_missing.push("pH值缺失，使用默认值7.0".to_string());
        7.0
    } else {
        ph
    };

    let ca_ppm = if ca_ppm.is_nan() || ca_ppm < 0.0 {
        data_missing.push("钙离子浓度缺失，使用默认值100.0ppm".to_string());
        100.0
    } else {
        ca_ppm
    };

    let orp_mv = if orp_mv.is_nan() || orp_mv.abs() > 2000.0 {
        data_missing.push("ORP值缺失，使用默认值0.0mV".to_string());
        0.0
    } else {
        orp_mv
    };

    let ambient_temp_c = if ambient_temp_c.is_nan() {
        data_missing.push("环境温度缺失，使用默认值20.0℃".to_string());
        20.0
    } else {
        ambient_temp_c
    };

    let ambient_rh_pct = if ambient_rh_pct.is_nan() || ambient_rh_pct <= 0.0 {
        data_missing.push("环境湿度缺失，使用默认值60.0%".to_string());
        60.0
    } else {
        ambient_rh_pct
    };

    let ph_condition = classify_ph_condition(ph);
    let ca_condition = classify_ca_condition(ca_ppm);
    let redox_condition = classify_origination(orp_mv);

    let mut decision_path: Vec<String> = Vec::new();
    decision_path.push(format!("步骤1: pH条件判定 = {} (pH={:.2})", ph_condition, ph));
    decision_path.push(format!(
        "步骤2: 钙离子浓度判定 = {} (Ca²⁺={:.1}ppm)",
        ca_condition, ca_ppm
    ));
    decision_path.push(format!(
        "步骤3: 氧化还原条件判定 = {} (ORP={:.0}mV)",
        redox_condition, orp_mv
    ));
    decision_path.push(format!(
        "步骤4: linfa决策树森林投票 ({}棵树，{}数据项缺失)",
        NUM_RF_TREES,
        data_missing.len()
    ));

    let forest = train_forest();
    let (best_class, votes) = predict_with_forest(&forest, ph, ca_ppm);

    let moisturizer = CLASS_LABELS[best_class].to_string();
    let moisturizer_zh = CLASS_LABELS_ZH[best_class].to_string();
    let concentration = CLASS_CONCENTRATIONS[best_class];
    let base_effectiveness = CLASS_EFFECTIVENESS[best_class];

    let top_votes = votes.get(&best_class).copied().unwrap_or(0) as f64;
    let total_votes = NUM_RF_TREES as f64;
    let raw_confidence = top_votes / total_votes;

    let missing_penalty = 1.0 - (data_missing.len() as f64 * 0.1);
    let orp_stability = if orp_mv > 200.0 || orp_mv < -200.0 {
        0.85
    } else {
        1.0
    };
    let rh_stability = if ambient_rh_pct < 30.0 || ambient_rh_pct > 90.0 {
        0.9
    } else {
        1.0
    };
    let temp_stability = if ambient_temp_c > 30.0 {
        0.9
    } else {
        1.0
    };

    let final_confidence = (raw_confidence
        * missing_penalty.max(0.3)
        * orp_stability
        * rh_stability
        * temp_stability)
        .clamp(0.0, 1.0);

    let effectiveness = base_effectiveness * final_confidence;

    let mut vote_vec: Vec<(usize, usize)> = votes.into_iter().collect();
    vote_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let top_candidates: Vec<String> = vote_vec
        .iter()
        .take(3)
        .map(|(class, _)| CLASS_LABELS[*class].to_string())
        .collect();
    let top_candidates_zh: Vec<String> = vote_vec
        .iter()
        .take(3)
        .map(|(class, _)| CLASS_LABELS_ZH[*class].to_string())
        .collect();
    let candidate_confidences: Vec<f64> = vote_vec
        .iter()
        .take(3)
        .map(|(_, count)| *count as f64 / total_votes)
        .collect();

    let vote_counts: HashMap<String, usize> = vote_vec
        .iter()
        .map(|(class, count)| (CLASS_LABELS[*class].to_string(), *count))
        .collect();

    decision_path.push(format!(
        "步骤5: 森林投票结果: 主方案『{}』获票{}/{}，置信度={:.1}%",
        moisturizer_zh,
        vote_counts.get(&moisturizer).cloned().unwrap_or(0),
        NUM_RF_TREES,
        final_confidence * 100.0
    ));

    let application_method =
        if relic_category.contains("牙") || relic_category.contains("齿") {
            "局部棉签涂布法: 用洁净棉签蘸取保湿液，沿牙骨质-釉质分界轻轻涂布，避免渗入牙髓腔。每30分钟补涂一次。".to_string()
        } else if relic_category.contains("骨") && burial_depth_m < 0.5 {
            "喷淋-包裹法: 先用低压喷壶均匀喷洒保湿液，再用预先浸透保湿液的无纺土工布（2-3层）紧密包裹，外裹PVC膜密封。".to_string()
        } else {
            "浸入-逐层包裹法: 小件可直接浸入保湿液5-10秒；大件先喷洒后用3层保湿纱布包裹，外加PE膜密封，标注方向。".to_string()
        };

    let (need_neutralize, neutralizer) = match ph_condition {
        "EXTREMELY_ACIDIC" => (
            true,
            Some("磷酸缓冲生理盐水 (PBS) pH=7.2，使用前进行点滴中和试验".to_string()),
        ),
        "EXTREMELY_ALKALINE" => (
            true,
            Some("0.1M硼酸缓冲液或稀醋酸溶液(pH=5.5)，分次逐步中和".to_string()),
        ),
        _ => (false, None),
    };

    let mut secondary: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    match redox_condition {
        "HIGHLY_OXIDIZING" => {
            secondary.push("添加0.1%抗坏血酸（维生素C）作为临时抗氧化剂".to_string());
            warnings.push(
                "⚠️ 高氧化环境：出土后暴露于空气将加速氧化，建议1小时内完成保湿密封，避光保存".to_string(),
            );
        }
        "STRONGLY_REDUCING" => {
            secondary.push("添加0.05%百里酚作为微生物抑制剂".to_string());
            warnings.push(
                "⚠️ 强还原环境：出土后需缓慢氧化（预暴露12-24小时），防止硫化物快速氧化产生硫酸破坏"
                    .to_string(),
            );
        }
        _ => {}
    }

    if ambient_temp_c > 28.0 {
        secondary.push(format!(
            "环境温度{:.1}℃偏高，需配备冰袋或冷藏箱维持15-20℃",
            ambient_temp_c
        ));
        warnings.push(
            "⚠️ 高温环境：微生物活性加倍，需缩短现场到实验室的运输时间（≤6小时）".to_string(),
        );
    }
    if ambient_rh_pct < 40.0 {
        warnings.push(format!(
            "⚠️ 环境湿度仅{:.0}%，干燥风险高，需双层密封保湿",
            ambient_rh_pct
        ));
    }

    if final_confidence < 0.5 {
        warnings.push(format!(
            "⚠️ 预测置信度较低({:.1}%)，建议补充现场测试数据后重新计算",
            final_confidence * 100.0
        ));
    } else if final_confidence < 0.75 {
        secondary.push(format!(
            "💡 置信度{:.1}%，建议准备备选方案：{}",
            final_confidence * 100.0,
            top_candidates_zh
                .get(1)
                .cloned()
                .unwrap_or_else(|| "无".to_string())
        ));
    }

    let stabilization_hours = if need_neutralize { 4.0 } else { 1.0 };

    let mut materials = vec![
        ProtectionMaterial {
            name: moisturizer.clone(),
            name_zh: moisturizer_zh.clone(),
            quantity_estimate: format!(
                "约{} mL/件文物（根据尺寸调整）",
                if concentration < 50.0 { 200 } else { 500 }
            ),
            purpose: "主要保湿剂，维持骨角质水合状态，防止干燥开裂".to_string(),
            priority: "必要".to_string(),
        },
        ProtectionMaterial {
            name: "Nonwoven_Geotextile".to_string(),
            name_zh: "无纺土工布/医用纱布".to_string(),
            quantity_estimate: "3层包裹，约0.5㎡/件".to_string(),
            purpose: "保湿液载体，均匀接触文物表面，防止直接接触塑料膜".to_string(),
            priority: "必要".to_string(),
        },
        ProtectionMaterial {
            name: "PVC_or_PE_Film".to_string(),
            name_zh: "PVC/PE保鲜膜".to_string(),
            quantity_estimate: "双层密封，宽度≥30cm".to_string(),
            purpose: "密封保湿，防止水分蒸发，隔绝外部污染".to_string(),
            priority: "必要".to_string(),
        },
        ProtectionMaterial {
            name: "ABS_Support_Mesh".to_string(),
            name_zh: "ABS塑料支撑网格".to_string(),
            quantity_estimate: "根据文物尺寸定制".to_string(),
            purpose: "脆弱骨骼的物理支撑，防止运输途中破碎".to_string(),
            priority: "推荐".to_string(),
        },
    ];

    if need_neutralize {
        if let Some(n) = neutralizer.clone() {
            materials.insert(
                0,
                ProtectionMaterial {
                    name: "Neutralization_Buffer".to_string(),
                    name_zh: n,
                    quantity_estimate: "按需配制，先小面积测试".to_string(),
                    purpose: "先调节pH至中性范围，再进行保湿处理".to_string(),
                    priority: "必要".to_string(),
                },
            );
        }
    }

    let mut protocol = vec![
        "步骤1：现场拍照记录（出土状态、方向、颜色、附着物），采集保存环境传感器数据".to_string(),
        "步骤2：用软毛刷（尼龙毛）轻轻清除表面浮土，避免用力擦拭导致颗粒摩擦损伤".to_string(),
    ];

    if need_neutralize {
        protocol.push(
            "步骤3：pH中和 - 使用缓冲液进行点滴法局部测试，确认无不良反应后逐步扩大处理面积，监测pH变化"
                .to_string(),
        );
        protocol.push(
            "步骤4：保湿处理 - 按照推荐方法涂抹/喷洒保湿剂，静置5分钟使溶液充分渗透".to_string(),
        );
    } else {
        protocol.push(
            "步骤3：保湿处理 - 按照推荐方法涂抹/喷洒保湿剂，静置5分钟使溶液充分渗透".to_string(),
        );
    }

    protocol.extend(vec![
        "步骤5：包裹密封 - 内层用湿润无纺土工布3层紧密包裹（预留观察窗），外层用PVC膜双层密封，用记号笔标注文物编号、方向（上下）、处理日期".to_string(),
        "步骤6：物理支撑 - 对脆弱骨骼（如肋骨、指骨）加装ABS网格支撑，填充减震材料（泡沫塑料、气泡膜）".to_string(),
        format!("步骤7：环境控制 - 运输途中维持温度15-20℃{}，避免阳光直射和剧烈震动",
            if ambient_temp_c > 28.0 { "（冷藏箱+冰袋）" } else { "" }),
        "步骤8：现场→实验室交接 - 抵达实验室后立即拆除外层密封膜，检查保湿状态，转入恒湿柜（RH=55±5%）或进行下一步加固处理（如B72树脂渗透）".to_string(),
    ]);

    ProtectionRecommendation {
        primary_moisturizer: moisturizer,
        primary_moisturizer_zh: moisturizer_zh,
        concentration_pct: concentration,
        application_method,
        secondary_recommendations: secondary,
        ph_neutralization_required: need_neutralize,
        neutralization_agent: neutralizer,
        expected_effectiveness_score: effectiveness,
        estimated_stabilization_hours: stabilization_hours,
        warnings,
        decision_path,
        materials_needed: materials,
        step_by_step_protocol: protocol,
        prediction_confidence: final_confidence,
        top_candidates,
        top_candidates_zh,
        candidate_confidences,
        data_missing_flags: data_missing,
        tree_vote_counts: vote_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acidic_environment_recommends_peg() {
        let rec = recommend_temporary_protection(5.5, 60.0, 50.0, 20.0, 60.0, 1.0, "人骨");
        assert!(
            rec.primary_moisturizer.to_lowercase().contains("peg")
                || rec.secondary_recommendations.iter().any(|s| s.to_lowercase().contains("peg")),
            "acidic should recommend PEG: {}",
            rec.primary_moisturizer
        );
    }

    #[test]
    fn test_strongly_acidic_needs_neutralization() {
        let rec = recommend_temporary_protection(4.0, 80.0, 0.0, 22.0, 55.0, 1.0, "兽骨");
        assert!(rec.ph_neutralization_required);
        assert!(rec.neutralization_agent.is_some());
    }

    #[test]
    fn test_neutral_ph_no_neutralization() {
        let rec = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.5, "骨器");
        assert!(!rec.ph_neutralization_required);
    }

    #[test]
    fn test_ph_classification_full_range() {
        assert_eq!(classify_ph_condition(2.0), "EXTREMELY_ACIDIC");
        assert_eq!(classify_ph_condition(4.5), "HIGHLY_ACIDIC");
        assert_eq!(classify_ph_condition(5.5), "MODERATELY_ACIDIC");
        assert_eq!(classify_ph_condition(6.8), "NEUTRAL");
        assert_eq!(classify_ph_condition(8.0), "MODERATELY_ALKALINE");
        assert_eq!(classify_ph_condition(9.0), "HIGHLY_ALKALINE");
        assert_eq!(classify_ph_condition(10.5), "EXTREMELY_ALKALINE");
    }

    #[test]
    fn test_ca_classification_full_range() {
        assert_eq!(classify_ca_condition(5.0), "VERY_LOW_CA");
        assert_eq!(classify_ca_condition(40.0), "LOW_CA");
        assert_eq!(classify_ca_condition(100.0), "NORMAL_CA");
        assert_eq!(classify_ca_condition(250.0), "HIGH_CA");
        assert_eq!(classify_ca_condition(400.0), "VERY_HIGH_CA");
    }

    #[test]
    fn test_orp_classification() {
        assert_eq!(classify_origination(300.0), "HIGHLY_OXIDIZING");
        assert_eq!(classify_origination(100.0), "MODERATELY_OXIDIZING");
        assert_eq!(classify_origination(-50.0), "MODERATELY_REDUCING");
        assert_eq!(classify_origination(-200.0), "STRONGLY_REDUCING");
    }

    #[test]
    fn test_effectiveness_score_positive() {
        let cases = vec![(4.0, 50.0), (7.0, 100.0), (9.0, 200.0)];
        for (ph, ca) in cases {
            let rec = recommend_temporary_protection(ph, ca, 0.0, 20.0, 60.0, 1.0, "人骨");
            assert!(
                rec.expected_effectiveness_score > 0.0 && rec.expected_effectiveness_score <= 100.0,
                "score out of range: pH={}, Ca={}, score={}",
                ph, ca, rec.expected_effectiveness_score
            );
        }
    }

    #[test]
    fn test_decision_path_not_empty() {
        let rec = recommend_temporary_protection(6.5, 80.0, 50.0, 22.0, 65.0, 1.0, "人骨");
        assert!(!rec.decision_path.is_empty());
        assert!(rec.decision_path.len() >= 3);
    }

    #[test]
    fn test_step_by_step_protocol_has_steps() {
        let rec = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.5, "人骨");
        assert!(rec.step_by_step_protocol.len() >= 5);
    }

    #[test]
    fn test_data_missing_reduces_confidence() {
        let rec_complete = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        let rec_missing = recommend_temporary_protection(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0, "人骨");
        assert!(rec_missing.data_missing_flags.len() >= 4);
        assert!(rec_missing.prediction_confidence <= rec_complete.prediction_confidence);
    }

    #[test]
    fn test_random_forest_vote_counts_sum() {
        let rec = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        let total_votes: usize = rec.tree_vote_counts.values().sum();
        assert_eq!(total_votes, NUM_RF_TREES);
    }

    #[test]
    fn test_concentration_between_0_and_100() {
        let cases = vec![(5.0, 30.0), (7.0, 80.0), (9.0, 150.0)];
        for (ph, ca) in cases {
            let rec = recommend_temporary_protection(ph, ca, 0.0, 20.0, 60.0, 1.0, "人骨");
            assert!(rec.concentration_pct > 0.0 && rec.concentration_pct <= 100.0);
        }
    }

    #[test]
    fn test_stabilization_hours_positive() {
        let rec = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        assert!(rec.estimated_stabilization_hours > 0.0);
    }

    #[test]
    fn test_primary_moisturizer_zh_chinese() {
        let rec = recommend_temporary_protection(7.0, 100.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        assert!(!rec.primary_moisturizer_zh.is_empty());
        assert!(!rec.primary_moisturizer.is_empty());
    }

    #[test]
    fn test_alkaline_environment_handling() {
        let rec = recommend_temporary_protection(9.0, 150.0, -100.0, 18.0, 50.0, 2.0, "人骨");
        assert!(!rec.primary_moisturizer.is_empty());
        assert!(rec.expected_effectiveness_score > 0.0);
        assert!(!rec.step_by_step_protocol.is_empty());
    }

    #[test]
    fn test_extreme_acid_boundary_4_5() {
        let rec_mild_acid = recommend_temporary_protection(4.6, 50.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        let rec_strong_acid = recommend_temporary_protection(4.4, 50.0, 0.0, 20.0, 60.0, 1.0, "人骨");
        assert_ne!(
            rec_mild_acid.ph_neutralization_required,
            rec_strong_acid.ph_neutralization_required
        );
    }
}
