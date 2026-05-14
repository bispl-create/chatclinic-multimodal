import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Optional
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json


def compute_dice_2d(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """두 2D binary mask 간 Dice 계산"""
    intersection = np.sum(mask_a & mask_b)
    sum_ab = np.sum(mask_a) + np.sum(mask_b)
    if sum_ab == 0:
        return 1.0  # 둘 다 비어있으면 consistent
    return 2.0 * intersection / sum_ab


def compute_slice_consistency(seg: np.ndarray, labels: List[int]) -> Dict:
    """
    3D segmentation의 인접 슬라이스 간 consistency를 측정합니다.

    Args:
        seg: (D, H, W) 형태의 segmentation volume
        labels: 측정할 label ID 리스트

    Returns:
        label별 평균 inter-slice Dice (ISC)와 전체 평균
    """
    D = seg.shape[0]
    if D < 2:
        return {}

    results = {}

    for label in labels:
        binary = (seg == label)
        dice_list = []

        for t in range(D - 1):
            slice_t = binary[t]
            slice_t1 = binary[t + 1]

            # 둘 다 비어있으면 skip (장기가 없는 슬라이스)
            if np.sum(slice_t) == 0 and np.sum(slice_t1) == 0:
                continue

            dice_list.append(compute_dice_2d(slice_t, slice_t1))

        if len(dice_list) > 0:
            results[label] = {
                'mean_isc': float(np.mean(dice_list)),
                'min_isc': float(np.min(dice_list)),
                'std_isc': float(np.std(dice_list)),
                'num_slices': len(dice_list),
            }

    return results


def evaluate_slice_consistency_folder(
    prediction_folder: str,
    labels: List[int],
    label_names: Optional[Dict[int, str]] = None,
    file_ending: str = '.nii.gz',
    tag: str = 'pred',
) -> Dict:
    """
    폴더 내 모든 prediction 파일에 대해 slice consistency를 측정합니다.

    Args:
        prediction_folder: prediction nifti 파일들이 있는 폴더
        labels: 측정할 label ID 리스트
        label_names: label ID → 이름 매핑 (optional)
        file_ending: 파일 확장자
    """
    pred_files = subfiles(prediction_folder, suffix=file_ending, join=True)

    all_results = []
    label_aggregated = {label: [] for label in labels}

    for pred_file in pred_files:
        case_name = os.path.basename(pred_file).replace(file_ending, '')
        seg = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))

        case_result = compute_slice_consistency(seg, labels)

        for label in labels:
            if label in case_result:
                label_aggregated[label].append(case_result[label]['mean_isc'])

        all_results.append({
            'case': case_name,
            'metrics': case_result,
        })

    # 전체 평균
    mean_results = {}
    for label in labels:
        if len(label_aggregated[label]) > 0:
            name = label_names[label] if label_names and label in label_names else str(label)
            mean_results[name] = {
                'mean_isc': float(np.mean(label_aggregated[label])),
                'min_isc': float(np.min(label_aggregated[label])),
                'std_isc': float(np.std(label_aggregated[label])),
                'num_cases': len(label_aggregated[label]),
            }

    foreground_iscs = []
    for label in labels:
        if len(label_aggregated[label]) > 0:
            foreground_iscs.append(np.mean(label_aggregated[label]))

    summary = {
        'foreground_mean_isc': float(np.mean(foreground_iscs)) if foreground_iscs else 0.0,
        'per_label': mean_results,
        'per_case': all_results,
    }

    output_file = join(prediction_folder, f'slice_consistency_{tag}.json')
    save_json(summary, output_file, sort_keys=False)
    print(f"Results saved to {output_file}")
    print(f"Foreground mean ISC: {summary['foreground_mean_isc']:.4f}")
    for name, vals in mean_results.items():
        print(f"  {name}: ISC={vals['mean_isc']:.4f} (min={vals['min_isc']:.4f}, std={vals['std_isc']:.4f})")

    return summary


def compute_slice_dice_variation(pred: np.ndarray, gt: np.ndarray, labels: List[int]) -> Dict:
    """
    슬라이스별 Dice(pred_t, gt_t)를 계산하고 인접 슬라이스 간 변화율을 측정합니다.

    Returns:
        label별 per-slice Dice의 mean, std, max_drop (인접 슬라이스 간 최대 Dice 하락)
    """
    D = pred.shape[0]
    results = {}

    for label in labels:
        pred_binary = (pred == label)
        gt_binary = (gt == label)

        slice_dices = []
        for t in range(D):
            # GT에 장기가 있는 슬라이스만
            if np.sum(gt_binary[t]) == 0 and np.sum(pred_binary[t]) == 0:
                continue
            slice_dices.append(compute_dice_2d(pred_binary[t], gt_binary[t]))

        if len(slice_dices) < 2:
            continue

        # 인접 슬라이스 간 Dice 변화량
        diffs = [abs(slice_dices[i+1] - slice_dices[i]) for i in range(len(slice_dices) - 1)]
        # Dice 하락 (양수 = 떨어짐)
        drops = [slice_dices[i] - slice_dices[i+1] for i in range(len(slice_dices) - 1)]

        results[label] = {
            'mean_dice': float(np.mean(slice_dices)),
            'std_dice': float(np.std(slice_dices)),
            'mean_diff': float(np.mean(diffs)),
            'max_drop': float(np.max(drops)),
            'num_slices': len(slice_dices),
        }

    return results


def evaluate_slice_dice_variation_folder(
    prediction_folder: str,
    gt_folder: str,
    labels: List[int],
    label_names: Optional[Dict[int, str]] = None,
    file_ending: str = '.nii.gz',
    tag: str = 'variation',
) -> Dict:
    pred_files = sorted(subfiles(prediction_folder, suffix=file_ending, join=True))

    all_results = []
    label_agg = {label: {'std_dice': [], 'mean_diff': [], 'max_drop': []} for label in labels}

    for pred_file in pred_files:
        case_name = os.path.basename(pred_file).replace(file_ending, '')
        gt_file = join(gt_folder, os.path.basename(pred_file))
        if not os.path.exists(gt_file):
            continue

        pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        gt_seg = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

        case_result = compute_slice_dice_variation(pred_seg, gt_seg, labels)

        for label in labels:
            if label in case_result:
                label_agg[label]['std_dice'].append(case_result[label]['std_dice'])
                label_agg[label]['mean_diff'].append(case_result[label]['mean_diff'])
                label_agg[label]['max_drop'].append(case_result[label]['max_drop'])

        all_results.append({'case': case_name, 'metrics': case_result})

    mean_results = {}
    for label in labels:
        if len(label_agg[label]['std_dice']) > 0:
            name = label_names[label] if label_names and label in label_names else str(label)
            mean_results[name] = {
                'std_dice': float(np.mean(label_agg[label]['std_dice'])),
                'mean_diff': float(np.mean(label_agg[label]['mean_diff'])),
                'max_drop': float(np.mean(label_agg[label]['max_drop'])),
            }

    all_std = [v['std_dice'] for v in mean_results.values()]
    all_diff = [v['mean_diff'] for v in mean_results.values()]
    all_drop = [v['max_drop'] for v in mean_results.values()]

    summary = {
        'foreground_mean': {
            'std_dice': float(np.mean(all_std)) if all_std else 0,
            'mean_diff': float(np.mean(all_diff)) if all_diff else 0,
            'max_drop': float(np.mean(all_drop)) if all_drop else 0,
        },
        'per_label': mean_results,
        'per_case': all_results,
    }

    output_file = join(prediction_folder, f'slice_dice_variation_{tag}.json')
    save_json(summary, output_file, sort_keys=False)

    print(f"\n=== Slice Dice Variation ===")
    print(f"{'':>15s}  {'std':>7s}  {'mean_diff':>10s}  {'max_drop':>9s}")
    for name, vals in mean_results.items():
        print(f"{name:>15s}  {vals['std_dice']:>7.4f}  {vals['mean_diff']:>10.4f}  {vals['max_drop']:>9.4f}")
    fg = summary['foreground_mean']
    print(f"{'MEAN':>15s}  {fg['std_dice']:>7.4f}  {fg['mean_diff']:>10.4f}  {fg['max_drop']:>9.4f}")
    print(f"\nResults saved to {output_file}")

    return summary


def compute_isc_tracking(pred: np.ndarray, gt: np.ndarray, labels: List[int]) -> Dict:
    """
    Pred의 inter-slice Dice 변화 패턴이 GT의 패턴을 얼마나 따라가는지 측정합니다.

    delta_t_gt   = Dice(GT_t, GT_{t+1})
    delta_t_pred = Dice(Pred_t, Pred_{t+1})
    E = mean(|delta_t_pred - delta_t_gt|)
    """
    D = pred.shape[0]
    results = {}

    for label in labels:
        pred_binary = (pred == label)
        gt_binary = (gt == label)

        gt_iscs = []
        pred_iscs = []

        for t in range(D - 1):
            gt_t = gt_binary[t]
            gt_t1 = gt_binary[t + 1]

            # 둘 다 비어있으면 skip
            if np.sum(gt_t) == 0 and np.sum(gt_t1) == 0 and np.sum(pred_binary[t]) == 0 and np.sum(pred_binary[t + 1]) == 0:
                continue

            gt_iscs.append(compute_dice_2d(gt_t, gt_t1))
            pred_iscs.append(compute_dice_2d(pred_binary[t], pred_binary[t + 1]))

        if len(gt_iscs) < 2:
            continue

        diffs = [abs(pred_iscs[i] - gt_iscs[i]) for i in range(len(gt_iscs))]

        # correlation
        gt_arr = np.array(gt_iscs)
        pred_arr = np.array(pred_iscs)
        corr = float(np.corrcoef(gt_arr, pred_arr)[0, 1]) if np.std(gt_arr) > 0 and np.std(pred_arr) > 0 else 0.0

        results[label] = {
            'mean_isc_error': float(np.mean(diffs)),
            'max_isc_error': float(np.max(diffs)),
            'correlation': corr,
            'num_slices': len(gt_iscs),
        }

    return results


def evaluate_isc_tracking_folder(
    prediction_folder: str,
    gt_folder: str,
    labels: List[int],
    label_names: Optional[Dict[int, str]] = None,
    file_ending: str = '.nii.gz',
    tag: str = 'isc_track',
) -> Dict:
    pred_files = sorted(subfiles(prediction_folder, suffix=file_ending, join=True))

    all_results = []
    label_agg = {label: {'mean_err': [], 'max_err': [], 'corr': []} for label in labels}

    for pred_file in pred_files:
        case_name = os.path.basename(pred_file).replace(file_ending, '')
        gt_file = join(gt_folder, os.path.basename(pred_file))
        if not os.path.exists(gt_file):
            continue

        pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        gt_seg = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

        case_result = compute_isc_tracking(pred_seg, gt_seg, labels)

        for label in labels:
            if label in case_result:
                label_agg[label]['mean_err'].append(case_result[label]['mean_isc_error'])
                label_agg[label]['max_err'].append(case_result[label]['max_isc_error'])
                label_agg[label]['corr'].append(case_result[label]['correlation'])

        all_results.append({'case': case_name, 'metrics': case_result})

    mean_results = {}
    for label in labels:
        if len(label_agg[label]['mean_err']) > 0:
            name = label_names[label] if label_names and label in label_names else str(label)
            mean_results[name] = {
                'mean_isc_error': float(np.mean(label_agg[label]['mean_err'])),
                'max_isc_error': float(np.mean(label_agg[label]['max_err'])),
                'correlation': float(np.mean(label_agg[label]['corr'])),
            }

    all_err = [v['mean_isc_error'] for v in mean_results.values()]
    all_max = [v['max_isc_error'] for v in mean_results.values()]
    all_corr = [v['correlation'] for v in mean_results.values()]

    summary = {
        'foreground_mean': {
            'mean_isc_error': float(np.mean(all_err)) if all_err else 0,
            'max_isc_error': float(np.mean(all_max)) if all_max else 0,
            'correlation': float(np.mean(all_corr)) if all_corr else 0,
        },
        'per_label': mean_results,
        'per_case': all_results,
    }

    output_file = join(prediction_folder, f'isc_tracking_{tag}.json')
    save_json(summary, output_file, sort_keys=False)

    print(f"\n=== ISC Tracking (Pred vs GT inter-slice pattern) ===")
    print(f"{'':>15s}  {'mean_err':>9s}  {'max_err':>9s}  {'corr':>7s}")
    for name, vals in mean_results.items():
        print(f"{name:>15s}  {vals['mean_isc_error']:>9.4f}  {vals['max_isc_error']:>9.4f}  {vals['correlation']:>7.4f}")
    fg = summary['foreground_mean']
    print(f"{'MEAN':>15s}  {fg['mean_isc_error']:>9.4f}  {fg['max_isc_error']:>9.4f}  {fg['correlation']:>7.4f}")
    print(f"\nResults saved to {output_file}")

    return summary


def compute_area_change_consistency(pred: np.ndarray, gt: np.ndarray, labels: List[int], eps: float = 1.0) -> Dict:
    """
    슬라이스 간 면적 변화율이 GT와 얼마나 일치하는지 측정합니다.

    r_t_gt   = |A_{t+1}^gt - A_t^gt| / (A_t^gt + eps)
    r_t_pred = |A_{t+1}^pred - A_t^pred| / (A_t^pred + eps)
    E_area   = mean(|r_t_pred - r_t_gt|)
    """
    D = pred.shape[0]
    results = {}

    for label in labels:
        pred_binary = (pred == label)
        gt_binary = (gt == label)

        # 슬라이스별 면적
        pred_areas = [float(np.sum(pred_binary[t])) for t in range(D)]
        gt_areas = [float(np.sum(gt_binary[t])) for t in range(D)]

        rate_diffs = []
        for t in range(D - 1):
            # GT에 장기가 없는 구간 skip
            if gt_areas[t] == 0 and gt_areas[t + 1] == 0:
                continue

            gt_denom = max(gt_areas[t], gt_areas[t + 1]) + eps
            pred_denom = max(pred_areas[t], pred_areas[t + 1]) + eps
            r_gt = abs(gt_areas[t + 1] - gt_areas[t]) / gt_denom
            r_pred = abs(pred_areas[t + 1] - pred_areas[t]) / pred_denom

            rate_diffs.append(abs(r_pred - r_gt))

        if len(rate_diffs) > 0:
            results[label] = {
                'mean_area_error': float(np.mean(rate_diffs)),
                'max_area_error': float(np.max(rate_diffs)),
                'std_area_error': float(np.std(rate_diffs)),
                'num_slices': len(rate_diffs),
            }

    return results


def evaluate_area_change_folder(
    prediction_folder: str,
    gt_folder: str,
    labels: List[int],
    label_names: Optional[Dict[int, str]] = None,
    file_ending: str = '.nii.gz',
    tag: str = 'area',
) -> Dict:
    pred_files = sorted(subfiles(prediction_folder, suffix=file_ending, join=True))

    all_results = []
    label_agg = {label: {'mean': [], 'max': []} for label in labels}

    for pred_file in pred_files:
        case_name = os.path.basename(pred_file).replace(file_ending, '')
        gt_file = join(gt_folder, os.path.basename(pred_file))
        if not os.path.exists(gt_file):
            continue

        pred_seg = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        gt_seg = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

        case_result = compute_area_change_consistency(pred_seg, gt_seg, labels)

        for label in labels:
            if label in case_result:
                label_agg[label]['mean'].append(case_result[label]['mean_area_error'])
                label_agg[label]['max'].append(case_result[label]['max_area_error'])

        all_results.append({'case': case_name, 'metrics': case_result})

    mean_results = {}
    for label in labels:
        if len(label_agg[label]['mean']) > 0:
            name = label_names[label] if label_names and label in label_names else str(label)
            mean_results[name] = {
                'mean_area_error': float(np.mean(label_agg[label]['mean'])),
                'max_area_error': float(np.mean(label_agg[label]['max'])),
            }

    all_mean = [v['mean_area_error'] for v in mean_results.values()]
    all_max = [v['max_area_error'] for v in mean_results.values()]

    summary = {
        'foreground_mean': {
            'mean_area_error': float(np.mean(all_mean)) if all_mean else 0,
            'max_area_error': float(np.mean(all_max)) if all_max else 0,
        },
        'per_label': mean_results,
        'per_case': all_results,
    }

    output_file = join(prediction_folder, f'area_change_{tag}.json')
    save_json(summary, output_file, sort_keys=False)

    print(f"\n=== Area Change Consistency ===")
    print(f"{'':>15s}  {'mean_err':>9s}  {'max_err':>9s}")
    for name, vals in mean_results.items():
        print(f"{name:>15s}  {vals['mean_area_error']:>9.4f}  {vals['max_area_error']:>9.4f}")
    fg = summary['foreground_mean']
    print(f"{'MEAN':>15s}  {fg['mean_area_error']:>9.4f}  {fg['max_area_error']:>9.4f}")
    print(f"\nResults saved to {output_file}")

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_folder', type=str)
    parser.add_argument('--gt_folder', type=str, default=None, help='GT folder for variation mode')
    parser.add_argument('--mode', type=str, default='consistency', choices=['consistency', 'variation', 'area', 'isc_track'],
                        help='consistency: inter-slice ISC, variation: per-slice Dice variation, area: area change, isc_track: ISC pattern tracking')
    parser.add_argument('--tag', type=str, default='pred', help='Output filename tag (e.g. pred, gt)')
    parser.add_argument('--labels', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    args = parser.parse_args()

    AMOS_LABEL_NAMES = {
        1: 'spleen', 2: 'R.kidney', 3: 'L.kidney', 4: 'gallbladder',
        5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
        9: 'postcava', 10: 'pancreas', 11: 'R.adrenal', 12: 'L.adrenal',
        13: 'duodenum', 14: 'bladder', 15: 'prostate',
    }

    if args.mode == 'consistency':
        evaluate_slice_consistency_folder(
            args.prediction_folder,
            labels=args.labels,
            label_names=AMOS_LABEL_NAMES,
            tag=args.tag,
        )
    elif args.mode == 'variation':
        assert args.gt_folder is not None, "--gt_folder required for variation mode"
        evaluate_slice_dice_variation_folder(
            args.prediction_folder,
            args.gt_folder,
            labels=args.labels,
            label_names=AMOS_LABEL_NAMES,
            tag=args.tag,
        )
    elif args.mode == 'isc_track':
        assert args.gt_folder is not None, "--gt_folder required for isc_track mode"
        evaluate_isc_tracking_folder(
            args.prediction_folder,
            args.gt_folder,
            labels=args.labels,
            label_names=AMOS_LABEL_NAMES,
            tag=args.tag,
        )
    elif args.mode == 'area':
        assert args.gt_folder is not None, "--gt_folder required for area mode"
        evaluate_area_change_folder(
            args.prediction_folder,
            args.gt_folder,
            labels=args.labels,
            label_names=AMOS_LABEL_NAMES,
            tag=args.tag,
        )
