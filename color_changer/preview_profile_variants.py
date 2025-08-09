#!/usr/bin/env python
"""
Preview script to sweep multiple color profile variants for a single base color.

Default use-case: Try several Black profile variants on 1.jpg, 2.jpg, 3.jpg
to quickly compare and pick the best base profile settings.
"""

import os
import sys
import argparse
import json
from copy import deepcopy
from datetime import datetime

# Import directly from local files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_changer.config.color_config import (
    PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH, COLOR_PROFILES
)
from color_changer.utils.color_utils import ColorUtils
from color_changer.utils.image_utils import ImageUtils
from color_changer.utils.preview_utils import (
    find_image_files, validate_image_list, get_valid_images_with_masks
)
from color_changer.core.color_transformer import ColorTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sweep multiple profile variants for a single color')
    parser.add_argument('--color', type=str, default='Black', help='Base color name (default: Black)')
    parser.add_argument('--images', nargs='+', default=['1.jpg', '2.jpg', '3.jpg'], help='Images to test (default: 1.jpg 2.jpg 3.jpg)')
    parser.add_argument('--images-dir', type=str, default=str(PREVIEW_IMAGES_DIR), help='Directory containing images')
    parser.add_argument('--results-dir', type=str, default=str(PREVIEW_RESULTS_DIR), help='Base results directory')
    parser.add_argument('--use-existing-masks', action='store_true', help='Use existing masks instead of generating new ones')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH), help='Model path (for mask generation if needed)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use for prediction (default: auto)')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization of results')
    return parser.parse_args()


def generate_black_variants(base_profile: dict):
    """Return a list of (variant_name, merged_profile_dict) for Black."""
    variants = []
    bp = deepcopy(base_profile)

    def merged(name, **over):
        prof = deepcopy(bp)
        for k, v in over.items():
            if isinstance(v, dict):
                prof.setdefault(k, {})
                prof[k].update(v)
            else:
                prof[k] = v
        variants.append((name, prof))

    # Baseline
    merged('v1_baseline')
    # Brighter shadows
    merged('v2_shadow_up', val={'shadow_gain': bp['val'].get('shadow_gain', 1.0) + 0.15})
    # Mid brighter
    merged('v3_mid_up', val={'mid_gain': bp['val'].get('mid_gain', 1.0) + 0.10})
    # Highlights a bit up + upper bound
    merged('v4_high_up', val={'highlight_gain': max(1.0, bp['val'].get('highlight_gain', 0.95) + 0.10), 'bounds': [bp['val']['bounds'][0], min(255, bp['val']['bounds'][1] + 10)]})
    # Slightly less saturation for perceived lightness
    merged('v5_sat_down', sat={'scale': max(1.0, bp['sat'].get('scale', 1.2) - 0.10)})
    # Slightly more saturation
    merged('v6_sat_up', sat={'scale': bp['sat'].get('scale', 1.2) + 0.15})
    # Stronger hue pull
    merged('v7_hue_weight', hue={'weight': bp['hue'].get('weight', 0.9) + 0.07})
    # Combined gentle brightening
    merged('v8_combined', hue={'weight': bp['hue'].get('weight', 0.9) + 0.03}, sat={'scale': max(1.0, bp['sat'].get('scale', 1.2) - 0.05)}, val={'shadow_gain': bp['val'].get('shadow_gain', 1.0) + 0.15, 'mid_gain': bp['val'].get('mid_gain', 1.0) + 0.10, 'highlight_gain': max(1.0, bp['val'].get('highlight_gain', 0.95) + 0.05), 'bounds': [bp['val']['bounds'][0], min(255, bp['val']['bounds'][1] + 15)]})

    # Darkening variants
    # Soft darkening
    merged('v9_dark_soft', val={
        'shadow_gain': max(0.70, bp['val'].get('shadow_gain', 1.0) - 0.10),
        'mid_gain': max(0.75, bp['val'].get('mid_gain', 1.0) - 0.12),
        'highlight_gain': max(0.70, bp['val'].get('highlight_gain', 0.95) - 0.10),
        'bounds': [bp['val']['bounds'][0], max(150, bp['val']['bounds'][1] - 10)]
    })
    # Stronger darkening
    merged('v10_dark_strong', val={
        'shadow_gain': max(0.60, bp['val'].get('shadow_gain', 1.0) - 0.20),
        'mid_gain': max(0.65, bp['val'].get('mid_gain', 1.0) - 0.18),
        'highlight_gain': max(0.60, bp['val'].get('highlight_gain', 0.95) - 0.20),
        'bounds': [bp['val']['bounds'][0], max(140, bp['val']['bounds'][1] - 20)]
    })
    # Slightly reduce saturation to enhance perceived darkness
    merged('v11_dark_sat_down', sat={
        'scale': max(0.9, bp['sat'].get('scale', 1.2) - 0.20),
        'min': max(0, bp['sat'].get('min', 0) - 10)
    }, val={
        'mid_gain': max(0.70, bp['val'].get('mid_gain', 1.0) - 0.10),
        'bounds': [bp['val']['bounds'][0], max(150, bp['val']['bounds'][1] - 10)]
    })

    # Deeper dark series based on v10_dark_strong idea (six steps)
    # Step 1: slight extra dark
    merged('v10d_1', val={
        'shadow_gain': max(0.55, bp['val'].get('shadow_gain', 1.0) - 0.25),
        'mid_gain': max(0.60, bp['val'].get('mid_gain', 1.0) - 0.23),
        'highlight_gain': max(0.55, bp['val'].get('highlight_gain', 0.95) - 0.25),
        'bounds': [bp['val']['bounds'][0], max(135, bp['val']['bounds'][1] - 25)]
    })
    # Step 2: more dark
    merged('v10d_2', val={
        'shadow_gain': max(0.50, bp['val'].get('shadow_gain', 1.0) - 0.30),
        'mid_gain': max(0.55, bp['val'].get('mid_gain', 1.0) - 0.28),
        'highlight_gain': max(0.50, bp['val'].get('highlight_gain', 0.95) - 0.30),
        'bounds': [bp['val']['bounds'][0], max(130, bp['val']['bounds'][1] - 30)]
    })
    # Step 3: include sat reduction to deepen
    merged('v10d_3', sat={
        'scale': max(0.85, bp['sat'].get('scale', 1.2) - 0.30)
    }, val={
        'shadow_gain': max(0.50, bp['val'].get('shadow_gain', 1.0) - 0.30),
        'mid_gain': max(0.55, bp['val'].get('mid_gain', 1.0) - 0.28),
        'highlight_gain': max(0.50, bp['val'].get('highlight_gain', 0.95) - 0.30),
        'bounds': [bp['val']['bounds'][0], max(125, bp['val']['bounds'][1] - 35)]
    })
    # Step 4: clamp highlights more
    merged('v10d_4', val={
        'shadow_gain': max(0.48, bp['val'].get('shadow_gain', 1.0) - 0.32),
        'mid_gain': max(0.53, bp['val'].get('mid_gain', 1.0) - 0.30),
        'highlight_gain': max(0.45, bp['val'].get('highlight_gain', 0.95) - 0.35),
        'bounds': [bp['val']['bounds'][0], max(120, bp['val']['bounds'][1] - 40)]
    })
    # Step 5: stronger overall dark
    merged('v10d_5', sat={
        'scale': max(0.80, bp['sat'].get('scale', 1.2) - 0.35)
    }, val={
        'shadow_gain': max(0.45, bp['val'].get('shadow_gain', 1.0) - 0.35),
        'mid_gain': max(0.50, bp['val'].get('mid_gain', 1.0) - 0.33),
        'highlight_gain': max(0.42, bp['val'].get('highlight_gain', 0.95) - 0.38),
        'bounds': [bp['val']['bounds'][0], max(115, bp['val']['bounds'][1] - 45)]
    })
    # Step 6: very deep dark
    merged('v10d_6', sat={
        'scale': max(0.75, bp['sat'].get('scale', 1.2) - 0.40)
    }, val={
        'shadow_gain': 0.40,
        'mid_gain': 0.45,
        'highlight_gain': 0.40,
        'bounds': [bp['val']['bounds'][0], 110]
    })

    # Extra steps targeting light hair (stronger mid/high darkening)
    merged('v10d_7', sat={
        'scale': 0.72
    }, val={
        'shadow_gain': 0.42,
        'mid_gain': 0.40,
        'highlight_gain': 0.35,
        'bounds': [bp['val']['bounds'][0], 102]
    })
    merged('v10d_8', sat={
        'scale': 0.70
    }, val={
        'shadow_gain': 0.40,
        'mid_gain': 0.38,
        'highlight_gain': 0.32,
        'bounds': [bp['val']['bounds'][0], 98]
    })
    merged('v10d_9', sat={
        'scale': 0.68
    }, val={
        'shadow_gain': 0.38,
        'mid_gain': 0.35,
        'highlight_gain': 0.30,
        'bounds': [bp['val']['bounds'][0], 94]
    })

    return variants


def generate_blonde_variants(base_profile: dict):
    """Variants for Blonde: focus on keeping highlights natural while lifting shadow/mid.
    Also provide darker variants for very dark source hair.
    """
    variants = []
    bp = deepcopy(base_profile)

    def merged(name, **over):
        prof = deepcopy(bp)
        for k, v in over.items():
            if isinstance(v, dict):
                prof.setdefault(k, {})
                prof[k].update(v)
            else:
                prof[k] = v
        variants.append((name, prof))

    # Baseline
    merged('v1_baseline')
    # Slight shadow/mid lift (better on medium/dark bases)
    merged('v2_shadow_mid_lift', val={'shadow_gain': bp['val'].get('shadow_gain', 1.2) + 0.10,
                                      'mid_gain': bp['val'].get('mid_gain', 1.10) + 0.10,
                                      'highlight_gain': max(0.90, bp['val'].get('highlight_gain', 0.95))})
    # Stronger mid lift, slightly reduce saturation to avoid yellow cast
    merged('v3_mid_lift_sat_down', sat={'scale': max(1.0, bp['sat'].get('scale', 1.15) - 0.10)},
           val={'mid_gain': bp['val'].get('mid_gain', 1.10) + 0.15,
                'highlight_gain': max(0.90, bp['val'].get('highlight_gain', 0.95))})
    # Highlight tamer (prevent blown highlights), slightly raise upper bound
    merged('v4_highlight_tamer', val={'highlight_gain': max(0.85, bp['val'].get('highlight_gain', 0.95) - 0.05),
                                      'bounds': [max(0, bp['val']['bounds'][0] - 5), min(255, bp['val']['bounds'][1] + 5)]})
    # Darker variants for very dark bases (reduce mid/high, reduce sat a touch)
    merged('v5_darker_soft', sat={'scale': max(1.0, bp['sat'].get('scale', 1.15) - 0.10)},
           val={'mid_gain': max(0.95, bp['val'].get('mid_gain', 1.10) - 0.10),
                'highlight_gain': max(0.85, bp['val'].get('highlight_gain', 0.95) - 0.10)})
    merged('v6_darker_strong', sat={'scale': max(0.95, bp['sat'].get('scale', 1.15) - 0.20)},
           val={'shadow_gain': max(0.95, bp['val'].get('shadow_gain', 1.20) - 0.10),
                'mid_gain': max(0.90, bp['val'].get('mid_gain', 1.10) - 0.15),
                'highlight_gain': max(0.80, bp['val'].get('highlight_gain', 0.95) - 0.15)})
    # Very dark base helper: more lift in shadows only, keep highlights tame
    merged('v7_shadow_focus', val={'shadow_gain': bp['val'].get('shadow_gain', 1.20) + 0.20,
                                   'mid_gain': max(1.00, bp['val'].get('mid_gain', 1.10)),
                                   'highlight_gain': max(0.90, bp['val'].get('highlight_gain', 0.95))})
    # Slight warmth (increase sat a bit), keep highlights controlled
    merged('v8_warmth', sat={'scale': bp['sat'].get('scale', 1.15) + 0.10},
           val={'highlight_gain': max(0.90, bp['val'].get('highlight_gain', 0.95) - 0.05)})

    return variants


def generate_brown_variants(base_profile: dict):
    """Variants for Brown: realistic warm/cool adjustments, lifts on dark base, tamer highlights.
    Provide darker and lighter sweeps to cover black→brown and blonde→brown transitions.
    """
    variants = []
    bp = deepcopy(base_profile)

    def merged(name, **over):
        prof = deepcopy(bp)
        for k, v in over.items():
            if isinstance(v, dict):
                prof.setdefault(k, {})
                prof[k].update(v)
            else:
                prof[k] = v
        variants.append((name, prof))

    # v1: baseline
    merged('v1_baseline')
    # v2: warm sat up, slight mid lift
    merged('v2_warm_mid', sat={'scale': bp['sat'].get('scale', 1.30) + 0.10},
           val={'mid_gain': bp['val'].get('mid_gain', 1.05) + 0.10, 'highlight_gain': max(0.90, bp['val'].get('highlight_gain', 0.90))})
    # v3: cool sat down, higher shadow lift (dark base helper)
    merged('v3_cool_shadow', sat={'scale': max(1.0, bp['sat'].get('scale', 1.30) - 0.10)},
           val={'shadow_gain': bp['val'].get('shadow_gain', 1.15) + 0.15, 'mid_gain': bp['val'].get('mid_gain', 1.05) + 0.05})
    # v4: highlight tamer
    merged('v4_highlight_tamer', val={'highlight_gain': max(0.85, bp['val'].get('highlight_gain', 0.90) - 0.05)})
    # v5: darker soft (good for blonde→brown without over-darkening)
    merged('v5_darker_soft', val={'mid_gain': max(0.98, bp['val'].get('mid_gain', 1.05) - 0.07),
                                  'highlight_gain': max(0.85, bp['val'].get('highlight_gain', 0.90) - 0.10)})
    # v6: darker strong (black→neutral brown)
    merged('v6_darker_strong', sat={'scale': max(1.0, bp['sat'].get('scale', 1.30) - 0.15)},
           val={'shadow_gain': max(0.92, bp['val'].get('shadow_gain', 1.15) - 0.15),
                'mid_gain': max(0.90, bp['val'].get('mid_gain', 1.05) - 0.12),
                'highlight_gain': max(0.80, bp['val'].get('highlight_gain', 0.90) - 0.15)})
    # v7: hue pull stronger for tone accuracy
    merged('v7_hue_weight', hue={'weight': bp['hue'].get('weight', 0.88) + 0.06})
    # v8: balanced bright (slightly higher bounds)
    merged('v8_balanced_bright', val={'shadow_gain': bp['val'].get('shadow_gain', 1.15) + 0.05,
                                      'mid_gain': bp['val'].get('mid_gain', 1.05) + 0.08,
                                      'highlight_gain': max(0.92, bp['val'].get('highlight_gain', 0.90)),
                                      'bounds': [bp['val']['bounds'][0], min(255, bp['val']['bounds'][1] + 10)]})

    return variants


def generate_auburn_variants(base_profile: dict):
    """Variants for Auburn: reddish-brown balance without drifting to copper.
    Focus on hue accuracy, moderate saturation, and controlled highlights.
    """
    variants = []
    bp = deepcopy(base_profile)

    def merged(name, **over):
        prof = deepcopy(bp)
        for k, v in over.items():
            if isinstance(v, dict):
                prof.setdefault(k, {})
                prof[k].update(v)
            else:
                prof[k] = v
        variants.append((name, prof))

    # v1: baseline
    merged('v1_baseline')
    # v2: stronger hue pull (avoid brown drift)
    merged('v2_hue_weight', hue={'weight': bp['hue'].get('weight', 0.85) + 0.08})
    # v3: reduce sat a touch to avoid copper
    merged('v3_sat_down', sat={'scale': max(1.0, bp['sat'].get('scale', 1.40) - 0.15)})
    # v4: shadow/mid warm lift, tame highlights
    merged('v4_shadow_mid', val={'shadow_gain': bp['val'].get('shadow_gain', 1.30) + 0.10,
                                 'mid_gain': bp['val'].get('mid_gain', 1.10) + 0.08,
                                 'highlight_gain': max(0.85, bp['val'].get('highlight_gain', 0.85))})
    # v5: darker soft for blonde base (less highlight)
    merged('v5_darker_soft', val={'mid_gain': max(1.00, bp['val'].get('mid_gain', 1.10) - 0.08),
                                  'highlight_gain': max(0.80, bp['val'].get('highlight_gain', 0.85) - 0.10)})
    # v6: darker strong for black base
    merged('v6_darker_strong', sat={'scale': max(1.0, bp['sat'].get('scale', 1.40) - 0.20)},
           val={'shadow_gain': max(0.95, bp['val'].get('shadow_gain', 1.30) - 0.20),
                'mid_gain': max(0.90, bp['val'].get('mid_gain', 1.10) - 0.15),
                'highlight_gain': max(0.75, bp['val'].get('highlight_gain', 0.85) - 0.10)})
    # v7: balanced bright (slightly higher bounds)
    merged('v7_balanced_bright', val={'shadow_gain': bp['val'].get('shadow_gain', 1.30) + 0.05,
                                      'mid_gain': bp['val'].get('mid_gain', 1.10) + 0.06,
                                      'bounds': [bp['val']['bounds'][0], min(255, bp['val']['bounds'][1] + 8)]})
    # v8: low-sat, strong hue (avoid copper/pink drift)
    merged('v8_low_sat_strong_hue', hue={'weight': bp['hue'].get('weight', 0.85) + 0.10},
           sat={'scale': max(1.0, bp['sat'].get('scale', 1.40) - 0.20)})

    return variants


def get_variants_for_color(color_label: str):
    base_profile = COLOR_PROFILES.get(color_label, COLOR_PROFILES['DEFAULT'])
    if color_label == 'Black':
        return generate_black_variants(base_profile)
    if color_label == 'Blonde':
        return generate_blonde_variants(base_profile)
    if color_label == 'Brown':
        return generate_brown_variants(base_profile)
    if color_label == 'Auburn':
        return generate_auburn_variants(base_profile)
    # Fallback: generic small sweeps for other colors
    variants = []
    bp = deepcopy(base_profile)
    variants.append(('v1_baseline', deepcopy(bp)))
    for i, delta in enumerate([(-0.1, 0.1), (0.1, 0.0), (0.0, 0.1)], start=2):
        sat_scale = max(1.0, bp['sat'].get('scale', 1.2) + delta[0])
        mid_up = bp['val'].get('mid_gain', 1.0) + delta[1]
        prof = deepcopy(bp)
        prof['sat']['scale'] = sat_scale
        prof['val']['mid_gain'] = mid_up
        variants.append((f'v{i}_sat_mid', prof))
    return variants


def main():
    args = parse_arguments()

    # Resolve color
    rgb, color_label = ColorUtils.find_color_by_name(args.color)
    if not color_label:
        print(f"Error: Color '{args.color}' not found.")
        sys.exit(1)

    # Prepare images
    if args.images:
        selected_images = validate_image_list(args.images, args.images_dir)
    else:
        selected_images = find_image_files(args.images_dir)
    if not selected_images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    # Masks
    valid_images, image_to_mask = get_valid_images_with_masks(
        selected_images, args.images_dir, args.use_existing_masks, args.model, args.device
    )
    if not valid_images:
        print("No valid images with masks found.")
        sys.exit(1)

    # Variants for this color
    variants = get_variants_for_color(color_label)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.results_dir, f"profile_variants_{color_label}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Keep original profile to restore afterwards
    original_profile = deepcopy(COLOR_PROFILES.get(color_label, {}))

    transformer = ColorTransformer()
    summary = {"color": color_label, "variants": []}
    aggregate_results = []  # For optional visualization

    try:
        for var_name, var_profile in variants:
            # Apply temporary override
            COLOR_PROFILES[color_label] = var_profile
            var_dir = os.path.join(out_dir, var_name)
            os.makedirs(var_dir, exist_ok=True)

            variant_entry = {"name": var_name, "profile": var_profile, "outputs": []}

            for img_path in valid_images:
                img_name = os.path.basename(img_path)
                base_name = os.path.splitext(img_name)[0]
                mask_path = image_to_mask[img_path]
                image = ImageUtils.load_image(img_path)
                mask = ImageUtils.load_image(mask_path, grayscale=True)
                if image is None or mask is None:
                    continue

                # Apply transformation with color label
                try:
                    result = transformer.change_hair_color(image, mask, color_label)
                    out_path = os.path.join(var_dir, f"{base_name}_{color_label.lower()}_{var_name}.png")
                    ImageUtils.save_image(result, out_path, convert_to_bgr=True)
                    variant_entry["outputs"].append({"image": img_name, "path": out_path})

                    # For visualization: group by image
                    # aggregate_results structure: [(img_name, [(label, path), ...]), ...]
                except Exception as e:
                    print(f"Failed on {img_name} with {var_name}: {e}")

            summary["variants"].append(variant_entry)

        # Build aggregate results for visualization
        # Re-scan saved files to fill structure expected by Visualizer
        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            img_entries = []
            base_name = os.path.splitext(img_name)[0]
            for var_name, _ in variants:
                var_dir = os.path.join(out_dir, var_name)
                out_path = os.path.join(var_dir, f"{base_name}_{color_label.lower()}_{var_name}.png")
                if os.path.exists(out_path):
                    img_entries.append((var_name, out_path))
            if img_entries:
                aggregate_results.append((img_name, img_entries))

        # Save summary JSON
        summary_path = os.path.join(out_dir, f"summary_{color_label}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")

        # Optional visualization
        if not args.no_visualization and aggregate_results:
            try:
                from color_changer.utils.visualization import Visualizer
                print("\nVisualizing variant results...")
                Visualizer.visualize_preview_results(aggregate_results)
            except Exception as e:
                print(f"Visualization failed: {e}")

    finally:
        # Restore original profile
        if original_profile:
            COLOR_PROFILES[color_label] = original_profile


if __name__ == "__main__":
    main()


