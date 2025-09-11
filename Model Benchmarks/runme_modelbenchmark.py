import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # Install HF to access the models

    

# === INFERENCE FUNCTION ===
def pick_phases_from_mat(mat_path, model_dict, preds_root):
    """
    Returns (p_time, s_time, cls_label or 'NA', model_display_name) or None if skipped.
    Skips when:
      - neither a local 'weights' file nor HF ('hf_repo','hf_filename') is provided
      - expected_seq_len is provided and doesn't equal total_samples (no padding allowed)
    If 'hf_repo'/'hf_filename' are provided, weights are fetched via huggingface_hub and cached
    under ~/.cache/huggingface/hub. Subsequent runs reuse the cached file automatically.
    Models are loaded with torch.jit.load (TorchScript format).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(preds_root, exist_ok=True)

    # ----- Load sample -----
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    eq = mat['EQ']
    record = eq.anEQ
    accel = record.Accel  # shape: [N, 3]

    
    sr = 100
    total_samples = int(accel.shape[0])
    duration = total_samples / sr
    record_name = os.path.basename(mat_path).replace('.mat', '')

    print(f"\nProcessing: {record_name}")
    print(f"  Samples: {total_samples} | SR: {sr} Hz | Duration: {duration:.2f}s")

    CLASS_LABELS = ["1N", "2NP", "2NPS"]

    # Helper: run a single TorchScript model on full signal (no padding, no windowing)
    def run_single_model(model_entry, expected_seq_len=None, model_type=None):
        """
        Returns (p_time, s_time, cls_label or 'NA', model_display_name) or None if skipped.
        Skips when:
          - neither a valid local 'weights' file nor HF ('hf_repo','hf_filename') is provided
          - expected_seq_len is provided and doesn't equal total_samples (no padding allowed)
        If 'hf_repo'/'hf_filename' are provided, weights are fetched via huggingface_hub and cached
        under ~/.cache/huggingface/hub. Subsequent runs reuse the cached file automatically.
        Models are loaded with torch.jit.load (TorchScript format).
        """
        model_name = model_entry.get("name", "Model")
        model_type = model_entry.get("model_type", "standard")

        # Strict length check for fixed-length models (your RESNET routing)
        if expected_seq_len is not None and expected_seq_len != total_samples:
            print(f"  [SKIP] {model_name}: expected {expected_seq_len} samples, got {total_samples}")
            return None

        # --- Resolve a local path OR an HF-cached path ---
        # Try local path first (relative to Models/ or absolute); then HF.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_models_root = os.path.join(base_dir, "Models")

        weights_cfg = model_entry.get("weights")  # may be None if using HF
        repo_id     = model_entry.get("hf_repo")
        hf_file     = model_entry.get("hf_filename")
        revision    = model_entry.get("hf_revision", None)

        weights_path = None

        # Local path branch
        if isinstance(weights_cfg, str):
            candidate = weights_cfg
            if not os.path.isabs(candidate):
                candidate = os.path.join(local_models_root, candidate)
            if os.path.isfile(candidate):
                weights_path = candidate

        # HF branch (only if no usable local file found)
        if weights_path is None and repo_id and hf_file:
            if hf_hub_download is None:
                print(f"  [SKIP] {model_name}: huggingface_hub not installed.")
                return None
            try:
                weights_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_file,
                    revision=revision,
                )
            except Exception as e:
                # Could be 404, permission error, network issue, etc, please check
                print(f"  [SKIP] {model_name}: failed to fetch from HF ({repo_id}/{hf_file}) -> {e}")
                return None


        if weights_path is None:
            print(f"  [SKIP] {model_name}: no valid local 'weights' or HF ('hf_repo','hf_filename').")
            return None

        # --- Load TorchScript model ---
        model = torch.jit.load(weights_path, map_location=device)
        model = model.to(device)
        model.eval()

        # Prepare input based on model type
        if model_type == "cnn_lstm":
            # CNN-LSTM specific preprocessing using your dataloader logic
            x = prepare_cnn_lstm_input(accel, model_entry)
        else:
            # Standard preprocessing (existing)
            x = torch.tensor(accel.T, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3, N]

        # Run inference
        with torch.no_grad():
            if model_type == "cnn_lstm":
                # CNN-LSTM models return predictions for each window, need to aggregate
                p_time, s_time, cls_label = run_cnn_lstm_inference(model, x, model_entry)
            else:
                # Standard model inference
                p_local, s_local, cls_logits = model(x)
                p_sec = float(p_local.item())
                s_sec = float(s_local.item())
                p_time = round(p_sec, 3)
                s_time = round(s_sec, 3)

                if isinstance(cls_logits, torch.Tensor) and cls_logits.ndim >= 2 and cls_logits.shape[0] >= 1:
                    # Standard classification for all models (including TimesNet)
                    cls_idx = int(torch.argmax(cls_logits, dim=1).item())
                    cls_label = CLASS_LABELS[cls_idx] if 0 <= cls_idx < len(CLASS_LABELS) else f"class{cls_idx}"
                else:
                    cls_label = "NA"

        return (p_time, s_time, cls_label, model_name)

    # Helper function for CNN-LSTM input preparation
    def prepare_cnn_lstm_input(accel, model_entry):
        """
        Prepare input for CNN-LSTM models using your windowing approach
        """
        # Get CNN-LSTM specific parameters
        window_size = model_entry.get("window_size", 900)  # samples per window
        hopping_size = model_entry.get("hopping_size", 300)  # hopping size
        
        # Apply your zero-center normalization
        accel_tensor = torch.tensor(accel, dtype=torch.float32)  # [N, 3]
        accel_tensor = zero_center_normalize(accel_tensor)  # [N, 3]
        
        # Apply windowing logic 
        windows = []
        num_steps = accel_tensor.shape[0] - window_size + 1
        
        for start_idx in range(0, num_steps, hopping_size):
            end_idx = start_idx + window_size
            window_data = accel_tensor[start_idx:end_idx, :]  # [window_size, 3]
            # Transpose to [3, window_size] to match dataloader format
            window_data = window_data.T  # [3, window_size]
            # Add the extra dimension: [3, 1, window_size]
            window_data = window_data.unsqueeze(1)  # [3, 1, window_size]
            windows.append(window_data)
        
        # Stack windows: [num_windows, 3, 1, window_size]
        if len(windows) == 0:
            # Handle edge case
            return torch.zeros(1, 3, 1, window_size, device=device)
        
        x = torch.stack(windows, dim=0).to(device)
        
        # Add batch dimension: [1, num_windows, 3, 1, window_size]
        x = x.unsqueeze(0)
        
        return x

    # Helper function for CNN-LSTM inference with confidence-based aggregation
    def run_cnn_lstm_inference(model, x, model_entry):
        """
        Run CNN-LSTM model and aggregate window predictions using confidence
        x shape: [1, num_windows, 3, 1, window_size]
        """
        # Get model predictions for all windows
        # model returns: [batch_size, num_windows, 4] where 4 = [p_idx, s_idx, p_confidence, s_confidence]
        # p_idx and s_idx are already GLOBAL indices (absolute sample positions)
        predictions = model(x)  # [1, num_windows, 4]
        
        # Extract values
        batch_size, num_windows, _ = predictions.shape
        predictions = predictions.squeeze(0)  # [num_windows, 4]
        
        # Parse predictions: [p_idx, s_idx, p_confidence, s_confidence]
        p_indices = predictions[:, 0]  # [num_windows] - global P wave indices
        s_indices = predictions[:, 1]  # [num_windows] - global S wave indices
        p_confidences = predictions[:, 2]  # [num_windows] - confidence for P wave
        s_confidences = predictions[:, 3]  # [num_windows] - confidence for S wave
        
        # Convert global indices to time (in seconds)
        sr = 100
        
        # Aggregate predictions using confidence-based approach
        p_time, s_time, cls_label = aggregate_cnn_lstm_predictions_global(
            p_indices, s_indices, p_confidences, s_confidences, sr
        )
        
        return p_time, s_time, cls_label

    # Helper function to aggregate CNN-LSTM predictions using global indices and confidence
    def aggregate_cnn_lstm_predictions_global(p_indices, s_indices, p_confidences, s_confidences, sr):
        """
        Aggregate predictions from multiple windows using confidence-based approach
        p_indices and s_indices are already GLOBAL indices (absolute sample positions)
        """
        # Confidence threshold - only use predictions with high confidence
        confidence_threshold = 0.8
        
        # Filter out invalid predictions (indices < 0)
        valid_p_mask = (p_indices >= 0) & (p_confidences > confidence_threshold)
        valid_s_mask = (s_indices >= 0) & (s_confidences > confidence_threshold)
        
        # Get valid high-confidence predictions
        valid_p_indices = p_indices[valid_p_mask]
        valid_s_indices = s_indices[valid_s_mask]
        valid_p_confidences = p_confidences[valid_p_mask]
        valid_s_confidences = s_confidences[valid_s_mask]
        
        # Aggregate P wave predictions - use max confidence prediction
        if len(valid_p_indices) > 0:
            # Use the prediction with highest confidence
            best_p_idx = torch.argmax(valid_p_confidences)
            p_time = round(float(valid_p_indices[best_p_idx].item() / sr), 3)
        else:
            p_time = None
        
        # Aggregate S wave predictions - use max confidence prediction
        if len(valid_s_indices) > 0:
            # Use the prediction with highest confidence
            best_s_idx = torch.argmax(valid_s_confidences)
            s_time = round(float(valid_s_indices[best_s_idx].item() / sr), 3)
        else:
            s_time = None
        
        # Determine classification based on confidence patterns
        p_detected = torch.any((p_indices >= 0) & (p_confidences > 0.5))
        s_detected = torch.any((s_indices >= 0) & (s_confidences > 0.5))
        
        if p_detected and s_detected:
            cls_label = "2NPS"
        elif p_detected:
            cls_label = "2NP"
        else:
            cls_label = "1N"
        
        return p_time, s_time, cls_label

    # Helper function for zero-center normalization (from your dataloader)
    def zero_center_normalize(data):
        # Zero-center each channel separately
        mean = data.mean(axis=-1, keepdims=True)
        normalized_data = data - mean
        return normalized_data

    # Determine sample length in seconds (int) for resnet routing
    sample_len_s = int(round(total_samples / sr))

    # Collect grouped keys so we can save under RESNET / CONV_LSTM / CNN_LSTM / etc.
    # Group name rule: "RESNET_15" -> "RESNET"; "RESNET30" -> "RESNET"; else group is the key itself.
    def group_name_from_key(k: str) -> str:
        ku = k.upper()
        if ku.startswith("RESNET"):
            return "RESNET"
        elif ku.startswith("TIMESNET"):
            return "TIMESNET"
        elif ku.startswith("CNN_LSTM"):
            return "CNN_LSTM"
        else:
            return k

    # First pass: RESNET special handling (choose correct 15/30/60/100 model by data length)
    if any(k.upper().startswith("RESNET") for k in model_dict.keys()):
        # Find the key that matches the length (e.g., contains '15' / '30' / '60' / '100')
        target_key = None
        for k in model_dict.keys():
            ku = k.upper()
            if not ku.startswith("RESNET"):
                continue
            if str(sample_len_s) in ku:
                target_key = k
                break

        if target_key is not None:
            res_entry = model_dict[target_key]
            res_group = group_name_from_key(target_key)
            out_dir = os.path.join(preds_root, res_group)
            os.makedirs(out_dir, exist_ok=True)

            result = run_single_model(
                model_entry=res_entry,
                expected_seq_len=total_samples  # must match exactly; no padding/resizing
            )
            if result is not None:
                p_time_cand, s_time_cand, cls_label, model_name = result
                
                # Apply masking by classification, ie. if model doesn't classify NPS, S prediction is not used
                print(cls_label)
                p_time, s_time, mask_note = _mask_regression_by_cls(cls_label, p_time_cand, s_time_cand)
            
                # ground truths from MAT (exported as p_true / s_true)
                p_gt = getattr(record, "p_true", -1)
                s_gt = getattr(record, "s_true", -1)
                try: p_gt = float(p_gt)
                except: p_gt = -1.0
                try: s_gt = float(s_gt)
                except: s_gt = -1.0
            
                # ---- Plot + Save ----
                time_axis = np.arange(total_samples) / sr
                plt.figure(figsize=(12, 5))
                for ch in range(3):
                    plt.plot(time_axis, accel[:, ch], label=f'Ch{ch}')
            
                # GT lines: solid
                if p_gt >= 0:
                    plt.axvline(p_gt, color="red",  linestyle="-",  linewidth=2, label=f"GT P: {p_gt:.2f}s")
                if s_gt >= 0:
                    plt.axvline(s_gt, color="blue", linestyle="-",  linewidth=2, label=f"GT S: {s_gt:.2f}s")
            
                # Predicted lines: dashed
                if p_time is not None and p_time >= 0:
                    plt.axvline(p_time, color="red",  linestyle="--", linewidth=2, label=f"Pred P: {p_time:.2f}s")
                if s_time is not None and s_time >= 0:
                    plt.axvline(s_time, color="blue", linestyle="--", linewidth=2, label=f"Pred S: {s_time:.2f}s")
            
                plt.title(f"{record_name} — {model_name} ({sample_len_s}s) — Type: {cls_label}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(True, axis='x', linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()
            
                safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
                plot_fname = f"{record_name}_{sample_len_s}s_{safe_model}_{cls_label}.png"
                plot_path = os.path.join(out_dir, plot_fname)
                plt.savefig(plot_path, dpi=150)
                plt.close()
            
                print(
                    f"  [RESNET] GT(P,S)=({p_gt},{s_gt}) "
                    f"Pred=({p_time},{s_time}) "
                    f"Type={cls_label} {'['+mask_note+']' if mask_note else ''} -> {plot_path}"
                )
        else:
            print(f"  [WARN] No matching RESNET model for {sample_len_s}s sample. Skipping RESNET.")

    # Second pass: TIMESNET special handling (same logic as RESNET)
    if any(k.upper().startswith("TIMESNET") for k in model_dict.keys()):
        # Find the key that matches the length (e.g., contains '15' / '30' / '60' / '100')
        target_key = None
        for k in model_dict.keys():
            ku = k.upper()
            if not ku.startswith("TIMESNET"):
                continue
            if str(sample_len_s) in ku:
                target_key = k
                break

        if target_key is not None:
            timesnet_entry = model_dict[target_key]
            timesnet_group = group_name_from_key(target_key)
            out_dir = os.path.join(preds_root, timesnet_group)
            os.makedirs(out_dir, exist_ok=True)

            result = run_single_model(
                model_entry=timesnet_entry,
                expected_seq_len=total_samples  # must match exactly; no padding/resizing
            )
            if result is not None:
                p_time_cand, s_time_cand, cls_label, model_name = result
                
                # Standard processing for all models (including TimesNet)
                p_time = p_time_cand
                s_time = s_time_cand
            
                # ground truths from MAT
                p_gt = getattr(record, "p_true", -1)
                s_gt = getattr(record, "s_true", -1)
                try: p_gt = float(p_gt)
                except: p_gt = -1.0
                try: s_gt = float(s_gt)
                except: s_gt = -1.0
            
                # ---- Plot + Save ----
                time_axis = np.arange(total_samples) / sr
                plt.figure(figsize=(12, 5))
                for ch in range(3):
                    plt.plot(time_axis, accel[:, ch], label=f'Ch{ch}')
            
                # GT lines: solid
                if p_gt >= 0:
                    plt.axvline(p_gt, color="red",  linestyle="-",  linewidth=2, label=f"GT P: {p_gt:.2f}s")
                if s_gt >= 0:
                    plt.axvline(s_gt, color="blue", linestyle="-",  linewidth=2, label=f"GT S: {s_gt:.2f}s")
            
                # Predicted lines: dashed
                if p_time is not None and p_time >= 0:
                    plt.axvline(p_time, color="red",  linestyle="--", linewidth=2, label=f"Pred P: {p_time:.2f}s")
                if s_time is not None and s_time >= 0:
                    plt.axvline(s_time, color="blue", linestyle="--", linewidth=2, label=f"Pred S: {s_time:.2f}s")
            
                plt.title(f"{record_name} — {model_name} ({sample_len_s}s) — Type: {cls_label}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(True, axis='x', linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()
            
                safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
                plot_fname = f"{record_name}_{sample_len_s}s_{safe_model}_{cls_label}.png"
                plot_path = os.path.join(out_dir, plot_fname)
                plt.savefig(plot_path, dpi=150)
                plt.close()
            
                # Handle -999 (None) values for display
                p_display = "None" if p_time < 0 else f"{p_time:.3f}"
                s_display = "None" if s_time < 0 else f"{s_time:.3f}"
                
                print(
                    f"  [TIMESNET] GT(P,S)=({p_gt:.3f},{s_gt:.3f}) "
                    f"Pred=({p_display},{s_display}) "
                    f"Type={cls_label} -> {plot_path}"
                )
        else:
            print(f"  [WARN] No matching TIMESNET model for {sample_len_s}s sample. Skipping TIMESNET.")

    # Third pass: CNN_LSTM special handling (same logic as RESNET and TIMESNET)
    if any(k.upper().startswith("CNN_LSTM") for k in model_dict.keys()):
        # Find the key that matches the length (e.g., contains '15' / '30' / '60' / '100')
        target_key = None
        for k in model_dict.keys():
            ku = k.upper()
            if not ku.startswith("CNN_LSTM"):
                continue
            if str(sample_len_s) in ku:
                target_key = k
                break

        if target_key is not None:
            cnn_lstm_entry = model_dict[target_key]
            cnn_lstm_group = group_name_from_key(target_key)
            out_dir = os.path.join(preds_root, cnn_lstm_group)
            os.makedirs(out_dir, exist_ok=True)

            result = run_single_model(
                model_entry=cnn_lstm_entry,
                expected_seq_len=None,  # CNN-LSTM accepts varying lengths
                model_type=cnn_lstm_entry.get("model_type", "cnn_lstm")
            )
            if result is not None:
                p_time_cand, s_time_cand, cls_label, model_name = result
                
                # Apply masking by classification, ie. if model doesn't classify NPS, S prediction is not used
                print(cls_label)
                p_time, s_time, mask_note = _mask_regression_by_cls(cls_label, p_time_cand, s_time_cand)
            
                # ground truths from MAT (exported as p_true / s_true)
                p_gt = getattr(record, "p_true", -1)
                s_gt = getattr(record, "s_true", -1)
                try: p_gt = float(p_gt)
                except: p_gt = -1.0
                try: s_gt = float(s_gt)
                except: s_gt = -1.0
            
                # ---- Plot + Save ----
                time_axis = np.arange(total_samples) / sr
                plt.figure(figsize=(12, 5))
                for ch in range(3):
                    plt.plot(time_axis, accel[:, ch], label=f'Ch{ch}')
            
                # GT lines: solid
                if p_gt >= 0:
                    plt.axvline(p_gt, color="red",  linestyle="-",  linewidth=2, label=f"GT P: {p_gt:.2f}s")
                if s_gt >= 0:
                    plt.axvline(s_gt, color="blue", linestyle="-",  linewidth=2, label=f"GT S: {s_gt:.2f}s")
            
                # Predicted lines: dashed
                if p_time is not None and p_time >= 0:
                    plt.axvline(p_time, color="red",  linestyle="--", linewidth=2, label=f"Pred P: {p_time:.2f}s")
                if s_time is not None and s_time >= 0:
                    plt.axvline(s_time, color="blue", linestyle="--", linewidth=2, label=f"Pred S: {s_time:.2f}s")
            
                plt.title(f"{record_name} — {model_name} ({sample_len_s}s) — Type: {cls_label}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(True, axis='x', linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()
            
                safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
                plot_fname = f"{record_name}_{sample_len_s}s_{safe_model}_{cls_label}.png"
                plot_path = os.path.join(out_dir, plot_fname)
                plt.savefig(plot_path, dpi=150)
                plt.close()
            
                print(
                    f"  [CNN_LSTM] GT(P,S)=({p_gt},{s_gt}) "
                    f"Pred=({p_time},{s_time}) "
                    f"Type={cls_label} {'['+mask_note+']' if mask_note else ''} -> {plot_path}"
                )
        else:
            print(f"  [WARN] No matching CNN_LSTM model for {sample_len_s}s sample. Skipping CNN_LSTM.")

    # Fourth pass: all OTHER models (e.g., CONV_LSTM, etc.)
    for key, entry in model_dict.items():
        ku = key.upper()
        if ku.startswith("RESNET") or ku.startswith("TIMESNET") or ku.startswith("CNN_LSTM"):
            continue  # already handled above

        group = group_name_from_key(key)
        out_dir = os.path.join(preds_root, group)
        os.makedirs(out_dir, exist_ok=True)

        result = run_single_model(
            model_entry=entry,
            expected_seq_len=None,  # these accept varying lengths; no strict seq-length check
            model_type=entry.get("model_type", "standard")
        )
        if result is None:
            continue

        p_time_cand, s_time_cand, cls_label, model_name = result
        
        # Apply masking by classification, ie. if model doesn't classify NPS, S prediction is not used
        print(cls_label)
        p_time, s_time, mask_note = _mask_regression_by_cls(cls_label, p_time_cand, s_time_cand)

        # ---- Plot + Save ----
        # ground truths from MAT (exported as p_true / s_true)
        p_gt = getattr(record, "p_true", -1)
        s_gt = getattr(record, "s_true", -1)
        try:
            p_gt = float(p_gt)
        except Exception:
            p_gt = -1.0
        try:
            s_gt = float(s_gt)
        except Exception:
            s_gt = -1.0
        
        time_axis = np.arange(total_samples) / sr
        plt.figure(figsize=(12, 5))
        for ch in range(3):
            plt.plot(time_axis, accel[:, ch], label=f'Ch{ch}')
        
        # GT lines: solid
        if p_gt is not None and p_gt >= 0:
            plt.axvline(p_gt, color="red",   linestyle="-",  linewidth=2, label=f"GT P: {p_gt:.2f}s")
        if s_gt is not None and s_gt >= 0:
            plt.axvline(s_gt, color="blue",  linestyle="-",  linewidth=2, label=f"GT S: {s_gt:.2f}s")
        
        # Predicted lines: dashed
        if p_time is not None and p_time >= 0:
            plt.axvline(p_time, color="red",   linestyle="--", linewidth=2, label=f"Pred P: {p_time:.2f}s")
        if s_time is not None and s_time >= 0:
            plt.axvline(s_time, color="blue",  linestyle="--", linewidth=2, label=f"Pred S: {s_time:.2f}s")
        
        plt.title(f"{model_name} — Type: {cls_label} — {record_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
        plot_fname = f"{record_name}_{safe_model}_{cls_label}.png"
        plot_path = os.path.join(out_dir, plot_fname)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  [{group}] GT(P,S)=({p_gt},{s_gt}) Pred(P,S)=({p_time},{s_time}) Type={cls_label} -> {plot_path}")


def _mask_regression_by_cls(cls_label: str, p_time, s_time):
    """
    Returns (p_plot, s_plot, note).
    - '1N'  (Noise):      suppress both P and S
    - '2NP' (Noise+P):    keep P, suppress S
    - '2NPS' (Noise+P+S): keep both
    Unknown labels -> keep both.
    """
    l = (cls_label or "").upper()
    p_ok, s_ok, note = True, True, None
    if l == "1N":
        p_ok, s_ok = False, False
        note = "masked: classified as Noise (1N)"
    elif l == "2NP":
        p_ok, s_ok = True, False
        note = "masked: classified as NP (2NP); S suppressed"
    # else '2NPS' or unknown -> keep both
    return (p_time if p_ok else None, s_time if s_ok else None, note)



# === BATCH EXECUTION (MAIN) ===

if __name__ == "__main__":

    base_dir   = os.path.dirname(os.path.abspath(__file__))  # folder of runme_modelbenchmark.py
    data_root  = os.path.join(base_dir, "Augmented Sample Data")
    models_root= os.path.join(base_dir, "Models")
    preds_root = os.path.join(base_dir, "Predictions")
    os.makedirs(preds_root, exist_ok=True)
    
    print(f"Data root:   {data_root}")
    print(f"Models root: {models_root}")
    print(f"Preds root:  {preds_root}")
    

    # Register models here
    model_dict = {

        "RESNET15":  {"name": "ResNet1D_15",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "resnet_15.torchscript"},
        "RESNET30":  {"name": "ResNet1D_30",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "resnet_30.torchscript"},
        "RESNET60":  {"name": "ResNet1D_60",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "resnet_60.torchscript"},
        "RESNET100": {"name": "ResNet1D_100", "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "resnet_100.torchscript"},
    
        "TIMESNET15":  {"name": "TimesNet_PS_15",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "timesnet_15.torchscript"},
        "TIMESNET30":  {"name": "TimesNet_PS_30",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "timesnet_30.torchscript"},
        "TIMESNET60":  {"name": "TimesNet_PS_60",  "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "timesnet_60.torchscript"},
        "TIMESNET100": {"name": "TimesNet_PS_100", "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", "hf_filename": "timesnet_100.torchscript"},
       
        # CNN-LSTM models with your custom preprocessing
        "CNN_LSTM_15": {
            "name": "CNN_LSTM_15s", 
            "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", 
            "hf_filename": "cnn_lstm_15.torchscript",
            "model_type": "cnn_lstm",
            "window_size": 900,  # 9 seconds * 100Hz
            "hopping_size": 300,  # 3 seconds * 100Hz
        },
        "CNN_LSTM_30": {
            "name": "CNN_LSTM_30s", 
            "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", 
            "hf_filename": "cnn_lstm_30.torchscript",
            "model_type": "cnn_lstm",
            "window_size": 900,  # 9 seconds * 100Hz
            "hopping_size": 300,  # 3 seconds * 100Hz
        },
        "CNN_LSTM_60": {
            "name": "CNN_LSTM_60s", 
            "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", 
            "hf_filename": "cnn_lstm_60.torchscript",
            "model_type": "cnn_lstm",
            "window_size": 900,  # 9 seconds * 100Hz
            "hopping_size": 300,  # 3 seconds * 100Hz
        },
        "CNN_LSTM_100": {
            "name": "CNN_LSTM_100s", 
            "hf_repo": "yek-models/P-and-S-Wave-Detection-in-Strong-Motion-Records-Comparative-Evaluation-of-Deep-Learning-Models", 
            "hf_filename": "cnn_lstm_100.torchscript",
            "model_type": "cnn_lstm",
            "window_size": 900,  # 9 seconds * 100Hz
            "hopping_size": 300,  # 3 seconds * 100Hz
        },
        
        # Register remaining models below
        #"CONV_LSTM": {"name": "CONV_LSTM_MODEL", "hf_repo": "jane-doe/repo-name", "hf_filename": "modelname.torchscript"},
    }


    # Process all .mat files directly in "Augmented Sample Data"
    mat_files = [f for f in os.listdir(data_root) if f.lower().endswith(".mat")]
    if not mat_files:
        print("No .mat files found in Augmented Sample Data.")
    else:
        for fname in mat_files:
            fpath = os.path.join(data_root, fname)
            print(f">>> Processing {fname}")
            pick_phases_from_mat(fpath, model_dict, preds_root)