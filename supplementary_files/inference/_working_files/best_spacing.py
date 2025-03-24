"""
This script aims at identifying the best spacing between samples used for inference, 
to maximize uniformity of coverage both at yearly and daily levels.

This means:
- The last sample is as close as possible to the last possible sample (no right gaps)
- All intra-day steps (4) are equally covered.
"""
from collections import Counter

max_valid_i = 1460 - 28 - 1 

spacings = dict()

for spacing in range(7, 200, 2):
    indices = list(range(0, max_valid_i, spacing))
    end_of_year_gap = max_valid_i - max(indices)
    c = Counter([i % 4 for i in indices])
    daily_coverage_uniformity = 0 if len(c) != 4 else min(c.values()) / max(c.values())
    spacings[spacing] = {
        "end_of_year_gap": end_of_year_gap,
        "daily_coverage_uniformity": daily_coverage_uniformity,
        "indices": indices
    }

# Identify min end_of_year_gap
min_end_of_year_gap = min(map(lambda x: spacings[x]["end_of_year_gap"], spacings))

# Keep only those that have this min level
spacings = {k: v for k, v in spacings.items() if v["end_of_year_gap"] == min_end_of_year_gap}

# Identify max daily_coverage_uniformity
max_daily_coverage_uniformity = max(map(lambda x: spacings[x]["daily_coverage_uniformity"], spacings))

# Order by daily_coverage_uniformity (descending)
spacings = dict(sorted(spacings.items(), key=lambda x: x[1]["daily_coverage_uniformity"], reverse=True))

# Print top 5
for c, (k, v) in enumerate(spacings.items()):
    if c >= 5:
        break
    print(f"Spacing: {k} (i.e. {len(v['indices'])} initial conditions), End of year gap: {v['end_of_year_gap']}, Daily coverage uniformity: {v['daily_coverage_uniformity']}")
