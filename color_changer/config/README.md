Configuration guide for hair color transformation

This document explains how the config in `color_config.py` influences the pipeline and how to tune it for different base hair types (dark vs light), and to prevent artifacts like pastel look or pixel stepping.

Where the config is used

- The unified handler reads a profile from `COLOR_PROFILES[<color>]` and applies:
  - Hue shift toward target (`hue.weight`)
  - Saturation scaling and optional approach to target S (`sat.*`)
  - Value (brightness) gains per region and clamped by bounds (`val.*`)
  - Optional corrections (highlight protection, smoothing, pink suppression, etc.)
- Lightening (separate block) can lift dark hair when the target is light.
- A universal `light_base` block (in `DEFAULT` and may be overridden per-color) softens transforms automatically for very light base hair.

Key knobs

1. hue

- `weight` (0..1): How strongly to move hue toward the target. Higher = stronger recolor. On very light hair, slightly lower values preserve natural variation. On very dark hair, values around 0.88–0.95 work well.
- Corrections:
  - `hue_band` [min,max]: Hard clamp of hue range; too tight can cause banding on light hair.
  - `hue_center` + `hue_center_weight`: Soft pull toward a center hue. Lower weight for smoother micro-textures.
  - `value_dependent_hue_center`: Stabilizes hue more in darker regions.

2. sat (saturation)

- `scale` (>1 increases color, <1 mutes). Dark hair tolerates higher scale; light hair requires moderation.
- `min`: Floor for saturation. High floors on light/desaturated hair can create pixel/steppy artifacts. Lower `min` for light hair.
- `max`: Cap for saturation to avoid neon/vivid outcomes on light hair.
- `approach_target_weight`: Extra nudge toward the target saturation. Lower values preserve texture and reduce stepping on light hair.
- `high_sat_boost`: Optional extra push when the target is very vivid.

3. val (value/brightness)

- `shadow_gain`, `mid_gain`, `highlight_gain`: Regional gains applied by V: <100 (shadow), 100–180 (mid), >180 (highlight). 1.0 keeps original; >1 brightens; <1 darkens.
- `bounds` [low, high]: Clamp after gains. Raising `high` prevents highlight dulling. Raising `low` lifts crushed blacks but reduces contrast.

4. corrections

- `highlight_protect`: Preserves specular highlights by blending hue with original, reducing saturation a bit, and capping V.
- `highlight_tamer`: Softer than protect; reduces harshness.
- `anti_pink`: Helpful for Blue/Purple to avoid pink drift.
- `desat_near_pink`: Reduces saturation around pink hues for realism.
- `post_smooth`, `bilateral_hue_smooth`: Recommended on light hair to reduce pixel-level stepping.

5. lightening

- Triggered only when base is dark and target is sufficiently light (controlled by `dark_thresh`, `light_thresh`).
- `shadow`, `mid`, `highlight`: Additive lifts by region; scaled by alpha.
- `desat`: Small desaturation during lift to avoid muddy casts.
- `upper_bound`: Max V after lift.

6. light_base (adaptive for very light hair)
   Defined in `DEFAULT.light_base` and optionally overridden per color. When mean hair V/255 exceeds `v_thresh` (≈0.68), the handler moderates transforms:

- Reduces hue weight (`hue_weight_mul`)
- Reduces saturation push (`sat_scale_mul`, lowers `sat.min` with `sat_min_reduce`, caps `sat.max` with `sat_max_cap`, reduces `approach_target_weight` with `sat_approach_mul`)
- Caps mid/high value gains and raises high bound (`val_highlight_gain_cap`, `val_mid_gain_cap`, `val_bounds_high`) to prevent dull highlights
- Forces smoothing (`force_post_smooth`, `force_bilateral_hue_smooth`) and enables `highlight_protect`
- Optionally clears `hue_band` which can cause stepping on light hair

Quick recipes

- Dark base → vivid color: increase `val.shadow_gain`/`mid_gain`, keep `bounds[1]` high (≥235), enable lightening with reasonable lifts. Use moderate `sat.scale` and avoid very high `sat.min` to keep texture.
- Light base → vivid color: lower `sat.scale`, keep `sat.min` low, cap `sat.max` ~210–230, reduce `approach_target_weight`. Keep `highlight_protect` on and smoothing enabled. Avoid tight `hue_band`.
- Pixel/steppy artifacts on light hair: lower `sat.min`, reduce `approach_target_weight`, enable smoothing, widen/remove `hue_band`.
- Dull highlights: raise `val.bounds[1]` and enable `highlight_protect` with `hp_v_cap` ≥ 240.

Troubleshooting examples

- Teal on blonde shows stepping: reduce `sat.min` (e.g., 150→100), reduce `sat.scale` (1.6→1.3), enable `bilateral_hue_smooth`, remove `hue_band`, cap `sat.max` ≤230.
- Purple on white looks pastel/flat: lower `sat.scale` and `approach_target_weight`, ensure `highlight_protect` is on, keep `val.highlight_gain` ≤1.0 and `bounds[1]` high (≥245), enable smoothing.

Notes

- All numeric changes are gentle; prefer incremental adjustments and compare against originals.
- The adaptive `light_base` moderation is non-destructive and only activates for light bases.
