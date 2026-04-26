import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEARNING_MAP = {
    0: 255,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 255,
}

# 19 valid classes + ignore(255)
PALETTE = np.array(
    [
        [245, 150, 100],
        [245, 230, 100],
        [150, 60, 30],
        [180, 30, 80],
        [255, 0, 0],
        [30, 30, 255],
        [200, 40, 255],
        [90, 30, 150],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [75, 0, 175],
        [0, 200, 255],
        [50, 120, 255],
        [0, 175, 0],
        [0, 60, 135],
        [80, 240, 150],
        [150, 240, 255],
        [0, 0, 255],
        [0, 0, 0],
    ],
    dtype=np.float32,
) / 255.0


def to_train_id(raw_labels: np.ndarray) -> np.ndarray:
    mapped = np.full(raw_labels.shape, 255, dtype=np.int32)
    for raw_id, train_id in LEARNING_MAP.items():
        mapped[raw_labels == raw_id] = train_id
    return mapped


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((labels.shape[0], 3), dtype=np.float32)
    ignore_mask = labels == 255
    valid_mask = ~ignore_mask
    rgb[ignore_mask] = PALETTE[19]
    if np.any(valid_mask):
        idx = np.clip(labels[valid_mask], 0, 18)
        rgb[valid_mask] = PALETTE[idx]
    return rgb


def read_stf_bin(bin_path: Path) -> np.ndarray:
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    return arr[:, :3]


def bev_panel(ax, xyz: np.ndarray, colors: np.ndarray, title: str, point_size: float) -> None:
    ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=point_size, marker='.', linewidths=0)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])


def load_weather_map(split_txt_path: Path) -> dict:
    weather_map = {}
    lines = split_txt_path.read_text().strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        stem, weather = line.strip().split(",")
        weather_map[stem] = weather
    return weather_map


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize SemanticSTF predictions.')
    parser.add_argument('--stf_root', type=str, required=True, help='SemanticSTF root path.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Folder with predicted .label files.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output folder for png files.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num_examples', type=int, default=6)
    parser.add_argument('--max_points', type=int, default=120000)
    parser.add_argument('--point_size', type=float, default=0.2)
    parser.add_argument(
        '--weather_types',
        type=str,
        default='snow,rain,dense_fog,light_fog',
        help='Comma-separated weather names. One example will be selected for each.',
    )
    parser.add_argument(
        '--split_txt',
        type=str,
        default='',
        help='Optional split annotation txt. If empty, use <stf_root>/<split>/<split>.txt',
    )
    args = parser.parse_args()

    split_dir = Path(args.stf_root) / args.split
    velo_dir = split_dir / 'velodyne'
    gt_dir = split_dir / 'labels'
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_txt = Path(args.split_txt) if args.split_txt else (split_dir / f'{args.split}.txt')

    pred_files = sorted(pred_dir.glob('*.label'))
    if len(pred_files) == 0:
        raise FileNotFoundError(f'No .label files found in {pred_dir}')

    if not split_txt.exists():
        raise FileNotFoundError(f'Weather split file not found: {split_txt}')

    weather_map = load_weather_map(split_txt)
    wanted_weathers = [w.strip() for w in args.weather_types.split(',') if w.strip()]

    pred_by_stem = {p.stem: p for p in pred_files}
    selected = []
    for weather in wanted_weathers:
        chosen_stem = None
        for stem, w in weather_map.items():
            if w == weather and stem in pred_by_stem:
                chosen_stem = stem
                break
        if chosen_stem is None:
            print(f'[Skip] No predicted frame found for weather: {weather}')
            continue
        selected.append((weather, pred_by_stem[chosen_stem]))

    # fallback: if weather-based selection is empty, keep old behavior
    if len(selected) == 0:
        selected = [('unknown', p) for p in pred_files[: args.num_examples]]

    print(f'Visualizing {len(selected)} weather-specific examples...')

    for i, (weather, pred_path) in enumerate(selected, start=1):
        stem = pred_path.stem
        bin_path = velo_dir / f'{stem}.bin'
        gt_path = gt_dir / f'{stem}.label'
        if not bin_path.exists() or not gt_path.exists():
            print(f'[Skip] Missing input file for {stem}')
            continue

        xyz = read_stf_bin(bin_path)
        gt_raw = np.fromfile(gt_path, dtype=np.int32).reshape(-1)
        pred = np.fromfile(pred_path, dtype=np.int32).reshape(-1)

        if xyz.shape[0] != gt_raw.shape[0] or xyz.shape[0] != pred.shape[0]:
            print(f'[Skip] Length mismatch {stem}: xyz={xyz.shape[0]}, gt={gt_raw.shape[0]}, pred={pred.shape[0]}')
            continue

        if xyz.shape[0] > args.max_points:
            idx = np.random.choice(xyz.shape[0], args.max_points, replace=False)
            xyz = xyz[idx]
            gt_raw = gt_raw[idx]
            pred = pred[idx]

        gt = to_train_id(gt_raw)

        gt_colors = labels_to_rgb(gt)
        pred_colors = labels_to_rgb(pred)

        # Save GT and prediction as two separate images (no collage).
        fig_gt = plt.figure(figsize=(8, 8))
        ax_gt = fig_gt.add_subplot(1, 1, 1)
        bev_panel(ax_gt, xyz, gt_colors, f'Ground Truth [{weather}]', args.point_size)
        fig_gt.tight_layout()
        out_gt = out_dir / f'{i:02d}_{weather}_{stem}_gt.png'
        fig_gt.savefig(out_gt, dpi=220)
        plt.close(fig_gt)

        fig_pred = plt.figure(figsize=(8, 8))
        ax_pred = fig_pred.add_subplot(1, 1, 1)
        bev_panel(ax_pred, xyz, pred_colors, f'Prediction [{weather}]', args.point_size)
        fig_pred.tight_layout()
        out_pred = out_dir / f'{i:02d}_{weather}_{stem}_pred.png'
        fig_pred.savefig(out_pred, dpi=220)
        plt.close(fig_pred)

        print(f'[Saved] {out_gt}')
        print(f'[Saved] {out_pred}')

    print(f'Done. Results are in: {out_dir}')


if __name__ == '__main__':
    main()
