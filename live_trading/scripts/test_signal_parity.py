"""
Signal Parity Test: Python vs Go inference

Compares XGBoost model predictions between:
1. Python (scikit-learn/xgboost)
2. Go (ONNX runtime)

To pass CP-001, signals must match on identical inputs.
"""

import json
import numpy as np
import xgboost as xgb
import onnxruntime as ort
from pathlib import Path


FEATURE_NAMES = [
    "total_intensity", "long_intensity", "short_intensity", "long_short_ratio",
    "above_below_ratio", "near_1pct_concentration", "near_2pct_concentration",
    "near_5pct_concentration", "largest_long_cluster_distance", "largest_short_cluster_distance",
    "largest_long_cluster_volume", "largest_short_cluster_volume",
    "top3_long_dist_1", "top3_long_dist_2", "top3_long_dist_3",
    "top3_short_dist_1", "top3_short_dist_2", "top3_short_dist_3",
    "entropy", "skewness",
    "return_1h", "return_6h", "return_12h", "return_24h",
    "volatility_6h", "volatility_24h", "atr_24h", "volume_ma_ratio",
    "wick_ratio_upper", "wick_ratio_lower", "price_position",
]


def generate_test_features(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    features = np.zeros((n_samples, 31))
    
    for i in range(n_samples):
        features[i, 0] = np.random.uniform(1e6, 1e9)
        features[i, 1] = features[i, 0] * np.random.uniform(0.4, 0.6)
        features[i, 2] = features[i, 0] - features[i, 1]
        features[i, 3] = features[i, 1] / max(features[i, 2], 1e-10)
        features[i, 4] = np.random.uniform(0.5, 2.0)
        features[i, 5:8] = np.random.uniform(0.0, 0.5, 3)
        features[i, 8:12] = np.random.uniform(0.01, 0.1, 4)
        features[i, 12:18] = np.random.uniform(0.01, 0.1, 6)
        features[i, 18] = np.random.uniform(1.0, 5.0)
        features[i, 19] = np.random.uniform(-0.05, 0.05)
        features[i, 20:24] = np.random.uniform(-0.05, 0.05, 4)
        features[i, 24:26] = np.random.uniform(0.001, 0.02, 2)
        features[i, 26] = np.random.uniform(0.005, 0.03)
        features[i, 27] = np.random.uniform(0.5, 2.0)
        features[i, 28:30] = np.random.uniform(0.0, 0.5, 2)
        features[i, 30] = np.random.uniform(0.0, 1.0)
    
    return features.astype(np.float32)


def test_xgboost_json(model_path: str, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = xgb.Booster()
    model.load_model(model_path)
    
    dmat = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
    probs = model.predict(dmat)
    
    labels = (probs > 0.5).astype(int)
    return labels, probs


def test_onnx(model_path: str, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(model_path)
    
    input_name = session.get_inputs()[0].name
    
    outputs = session.run(None, {input_name: features})
    
    labels = outputs[0].flatten()
    probs = outputs[1]
    
    return labels, probs


def main():
    base_path = Path(__file__).parent.parent
    json_path = base_path / "models" / "xgb_optuna_best.json"
    onnx_path = base_path / "models" / "xgb_optuna_best.onnx"
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return False
    
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found")
        return False
    
    print("Generating test features...")
    features = generate_test_features(n_samples=100)
    
    print(f"Testing XGBoost JSON model: {json_path}")
    xgb_labels, xgb_probs = test_xgboost_json(str(json_path), features)
    
    print(f"Testing ONNX model: {onnx_path}")
    onnx_labels, onnx_probs = test_onnx(str(onnx_path), features)
    
    label_match = np.sum(xgb_labels == onnx_labels)
    label_pct = label_match / len(xgb_labels) * 100
    
    onnx_probs_class1 = onnx_probs[:, 1] if len(onnx_probs.shape) > 1 else onnx_probs
    prob_diff = np.abs(xgb_probs - onnx_probs_class1)
    max_diff = np.max(prob_diff)
    mean_diff = np.mean(prob_diff)
    
    print("\n" + "="*60)
    print("SIGNAL PARITY TEST RESULTS")
    print("="*60)
    print(f"Samples tested:     {len(features)}")
    print(f"Label matches:      {label_match}/{len(features)} ({label_pct:.1f}%)")
    print(f"Max prob diff:      {max_diff:.6f}")
    print(f"Mean prob diff:     {mean_diff:.6f}")
    print("="*60)
    
    passed = label_pct >= 99.0 and max_diff < 0.01
    
    if passed:
        print("✅ CP-001 PASSED: Signal parity verified")
    else:
        print("❌ CP-001 FAILED: Signal mismatch detected")
        
        mismatches = np.where(xgb_labels != onnx_labels)[0]
        if len(mismatches) > 0:
            print(f"\nMismatched samples: {mismatches[:10]}...")
            for idx in mismatches[:5]:
                print(f"  Sample {idx}: XGB={xgb_labels[idx]} ({xgb_probs[idx]:.4f}), "
                      f"ONNX={onnx_labels[idx]} ({onnx_probs_class1[idx]:.4f})")
    
    results = {
        "samples": len(features),
        "label_matches": int(label_match),
        "label_match_pct": float(label_pct),
        "max_prob_diff": float(max_diff),
        "mean_prob_diff": float(mean_diff),
        "passed": bool(passed),
    }
    
    results_path = base_path / "checkpoints" / "signal_parity_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
