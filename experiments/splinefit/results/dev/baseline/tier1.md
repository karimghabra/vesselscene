# v0 tier 1 (run_experiment --tier 1, 2026-10-05)

```
status: ok
composite: 0.759238
graph: 0.608675
pos: 0.687964
width: 0.761496
explained: 0.782555
explained_junction: 0.781049
fast_composite: 0.676340
probe_score: 0.842137
probe_width: 2.5
probe_blur: 25.0
probe_contrast: 0.25
probe_corner: 1/1
probe_fork: 4/5
probe_crossing_angle: 30
probe_crossing_depth: 5.0
probe_crossing_width: 1/1
probe_t: 2/2
probe_gap: 2.0
probe_gap_truth: 2.0
probe_end: 2/2
probe_rbc: 2/2
probe_empty: 1/2 empty; max len 93 px
probe_mean_width: 0.7542
probe_mean_blur: 0.9166
probe_mean_contrast: 0.7870
probe_mean_corner: 0.7136
probe_mean_fork: 0.8643
probe_mean_crossing_angle: 0.8868
probe_mean_crossing_depth: 0.9367
probe_mean_crossing_width: 0.8858
probe_mean_t: 0.9735
probe_mean_gap: 0.7977
probe_mean_end: 0.9514
probe_mean_rbc: 0.9346
probe_mean_empty: 0.5350
seconds: 271.9
digest: a6bd0c1671b8bc41
```

| crop | kind | composite | graph | pos | width | explained | explained_junction | junction_bias | s |
|---|---|---|---|---|---|---|---|---|---|
| healthy_s000_480x768 [224, 32] | average | 0.772 | 0.680 | 0.815 | 0.840 | 0.934 | 0.950 | -0.187 | 22 |
| healthy_s000_480x768 [128, 128] | frame | 0.714 | 0.623 | 0.729 | 0.792 | 0.894 | 0.915 | -0.199 | 20 |
| healthy_s004_320x512 [64, 0] | average | 0.745 | 0.620 | 0.792 | 0.860 | 0.955 | 0.971 | -0.171 | 17 |
| healthy_s004_320x512 [224, 0] | frame | 0.640 | 0.561 | 0.606 | 0.750 | 0.799 | 0.806 | -0.430 | 18 |
| healthy_s007_480x768 [0, 0] | average | 0.704 | 0.681 | 0.762 | 0.720 | 0.701 | 0.561 | -0.569 | 17 |
| healthy_s007_480x768 [96, 192] | frame | 0.658 | 0.610 | 0.627 | 0.630 | 0.862 | 0.892 | -0.273 | 20 |
| healthy_s008_480x768 [0, 224] | average | 0.786 | 0.663 | 0.858 | 0.882 | 0.986 | 0.989 | -0.102 | 17 |
| healthy_s008_480x768 [96, 128] | frame | 0.826 | 0.728 | 0.859 | 0.931 | 0.983 | 0.988 | -0.078 | 15 |
| pathologic_s000_480x768 [0, 128] | average | 0.366 | 0.446 | 0.226 | 0.452 | 0.182 | 0.193 | -0.861 | 14 |
| pathologic_s000_480x768 [352, 96] | frame | 0.553 | 0.475 | 0.606 | 0.759 | 0.530 | 0.545 | -0.666 | 10 |
