"""
Common constants for hair color changing utilities.
"""

# Mask threshold used to decide hair region membership (normalized mask 0..1)
MASK_THRESHOLD: float = 0.1

# If hair coverage ratio (fraction of pixels over threshold) is below this,
# treat as effectively empty and skip processing
MIN_MASK_COVERAGE_RATIO: float = 0.005


