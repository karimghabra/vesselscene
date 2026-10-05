## probes: run experiments.splinefit.pipeline:annotate

probe_score 0.8421  digest 11bf4a16aaa51749  172 s  (34 probes)

| sweep | mean | threshold |
|---|---|---|
| width | 0.754 | 2.5 (smallest diameter detected) |
| blur | 0.917 | 25.0 (largest defocus detected with width err < 0.25) |
| contrast | 0.787 | 0.25 (smallest haematocrit detected) |
| corner | 0.714 | 1/1 (thin, defocused, low contrast detected) |
| fork | 0.864 | 4/5 (forks matched and typed 3-way) |
| crossing_angle | 0.887 | 30 (smallest angle typed a crossing) |
| crossing_depth | 0.937 | 5.0 (smallest depth gap typed a crossing) |
| crossing_width | 0.886 | 1/1 (unequal crossing typed a crossing) |
| t | 0.973 | 2/2 (T junctions matched and typed 3-way) |
| gap | 0.798 | 2.0 (smallest wall gap resolved (2 lines)) |
| end | 0.951 | 2/2 (ends traced without a false junction) |
| rbc | 0.935 | 2/2 (capillary detected) |
| empty | 0.535 | 1/2 empty; max len 93 px (nothing fitted) |
| gap (truth) | | 2.0 (smallest gap the observable truth resolves) |

**width** (x: diameter px)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d2.5 | 2.500 | average | 1.25/1.80/0.06 | 0.445 | 0.591 | 0.948 | 0.000 | - | 0.829 | 0.041 | 0.025 | - | - | 254.331 | None |
| line_d3 | 3.000 | average | 1.50/1.80/0.08 | 0.635 | 0.591 | 0.719 | 0.000 | - | 0.879 | 0.478 | 0.678 | - | - | 215.649 | None |
| line_d6 | 6.000 | average | 3.00/1.80/0.15 | 0.954 | 0.998 | 1.000 | 0.000 | - | 0.916 | 0.873 | 0.941 | - | - | 138.553 | None |
| line_d12 | 12.000 | average | 6.00/1.80/0.29 | 0.983 | 0.994 | 1.000 | 0.000 | - | 0.932 | 0.991 | 0.995 | - | - | 143.227 | None |

**blur** (x: focus offset d_c)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d6 | 0.000 | average | 3.00/1.80/0.15 | 0.954 | 0.998 | 1.000 | 0.000 | - | 0.916 | 0.873 | 0.941 | - | - | 138.553 | None |
| line_d6_f25 | 25.000 | average | 3.00/2.34/0.15 | 0.936 | 0.998 | 1.000 | 0.000 | - | 0.868 | 0.817 | 0.939 | - | - | 138.944 | None |
| line_d6_f50 | 50.000 | average | 3.00/3.50/0.15 | 0.897 | 0.996 | 1.000 | 0.000 | - | 0.720 | 0.740 | 0.934 | - | - | 137.833 | None |

**contrast** (x: haematocrit)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d6 | 1.000 | average | 3.00/1.80/0.15 | 0.954 | 0.998 | 1.000 | 0.000 | - | 0.916 | 0.873 | 0.941 | - | - | 138.553 | None |
| line_d6_h0.5 | 0.500 | average | 3.00/1.80/0.08 | 0.908 | 0.974 | 1.000 | 0.000 | - | 0.874 | 0.769 | 0.882 | - | - | 154.688 | None |
| line_d6_h0.25 | 0.250 | average | 3.00/1.80/0.04 | 0.666 | 0.587 | 0.733 | 0.000 | - | 0.737 | 0.864 | 0.636 | - | - | 218.817 | None |

**corner** (x: -)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d5_f25_h0.5 | d5 f25 h0.5 | average | 2.50/2.34/0.07 | 0.714 | 0.651 | 0.970 | 0.000 | - | 0.754 | 0.764 | 0.810 | - | - | 221.081 | None |

**fork** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fork_thin | thin r0=2 | average | 1.59/1.80/0.08 | 0.883 | 0.933 | 0.930 | 1.000 | - | 0.859 | 0.721 | 0.920 | 0.947 | -0.218 | 287.545 | None |
| fork_wide | wide r0=6 | average | 4.76/1.80/0.24 | 0.722 | 0.479 | 0.996 | 0.000 | - | 0.940 | 0.958 | 0.996 | 0.997 | -0.030 | 257.530 | None |
| fork_asym | asym 6>5.5+3.7 | average | 5.50/1.80/0.27 | 0.976 | 0.992 | 1.000 | 1.000 | - | 0.927 | 0.960 | 0.996 | 0.994 | -0.060 | 219.130 | None |
| fork_acute | acute 2x15deg | average | 3.17/1.80/0.16 | 0.885 | 0.869 | 1.000 | 0.667 | - | 0.946 | 0.783 | 0.977 | 0.987 | -0.045 | 244.938 | None |
| fork_thin_frame | thin frame | frame | 1.59/1.69/0.08 | 0.855 | 0.997 | 1.000 | 1.000 | - | 0.892 | 0.786 | 0.458 | 0.789 | -0.242 | 220.033 | None |

**crossing_angle** (x: angle deg)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a20 | 20 | average | 3.00/2.21/0.13 | 0.754 | 0.589 | 1.000 | 0.500 | 0.000 | 0.919 | 0.880 | 0.955 | 0.988 | -0.023 | 318.859 | None |
| cross_a30 | 30 | average | 3.00/2.21/0.13 | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.535 | 0.978 | 0.993 | -0.034 | 303.567 | None |
| cross_a45 | 45 | average | 3.00/2.21/0.13 | 0.895 | 1.000 | 1.000 | 1.000 | 1.000 | 0.906 | 0.485 | 0.981 | 0.994 | -0.055 | 288.719 | None |
| cross_a90 | 90 | average | 3.00/2.00/0.14 | 0.957 | 0.998 | 1.000 | 1.000 | 1.000 | 0.955 | 0.809 | 0.985 | 0.987 | -0.115 | 272.512 | None |
| cross_a45_frame | 45 frame | frame | 3.00/2.12/0.13 | 0.919 | 0.999 | 1.000 | 1.000 | 1.000 | 0.908 | 0.740 | 0.872 | 0.920 | 0.049 | 298.052 | None |

**crossing_depth** (x: depth gap d_c)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a45_dz5 | 5.000 | average | 3.00/1.83/0.15 | 0.949 | 0.999 | 1.000 | 1.000 | 1.000 | 0.928 | 0.778 | 0.988 | 0.995 | -0.042 | 296.041 | None |
| cross_a45_dz50 | 50.000 | average | 3.00/3.61/0.09 | 0.925 | 0.975 | 0.958 | 1.000 | 1.000 | 0.902 | 0.743 | 0.979 | 0.993 | -0.049 | 297.268 | None |

**crossing_width** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a45_wide_over_thin | 6 over 1.5 | average | 1.50/2.21/0.06 | 0.886 | 0.998 | 1.000 | 1.000 | 1.000 | 0.876 | 0.448 | 0.997 | 0.998 | -0.026 | 298.803 | None |

**t** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t_branch | T branch | average | 4.95/1.80/0.25 | 0.971 | 0.998 | 1.000 | 1.000 | - | 0.939 | 0.898 | 0.995 | 0.994 | -0.077 | 203.572 | None |
| pseudo_t | pseudo-T | average | 5.00/1.80/0.25 | 0.976 | 0.996 | 1.000 | 1.000 | - | 0.922 | 0.948 | 0.996 | 0.997 | -0.033 | 201.341 | None |

**gap** (x: wall gap px)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parallel_g1 | 1.000 | average | 2.00/1.80/0.11 | 0.712 | 0.991 | 0.978 | 0.000 | - | 0.393 | 0.000 | 0.904 | - | - | 135.943 | 1 |
| parallel_g2 | 2.000 | average | 2.00/1.80/0.11 | 0.663 | 0.664 | 0.815 | 0.000 | - | 0.286 | 0.814 | 0.885 | - | - | 269.750 | 2 |
| parallel_g3 | 3.000 | average | 2.00/1.80/0.11 | 0.915 | 0.999 | 1.000 | 0.000 | - | 0.923 | 0.689 | 0.879 | - | - | 268.847 | 2 |
| parallel_g5 | 5.000 | average | 2.00/1.80/0.11 | 0.901 | 0.999 | 1.000 | 0.000 | - | 0.929 | 0.630 | 0.853 | - | - | 268.179 | 2 |

**end** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| end_dive | dive | average | 2.50/2.43/0.14 | 0.938 | 0.958 | 1.000 | 0.000 | - | 0.895 | 0.885 | 0.976 | - | - | 107.602 | None |
| end_blind | blind end | average | 2.00/1.80/0.10 | 0.964 | 0.987 | 1.000 | 0.000 | - | 0.912 | 0.952 | 0.960 | - | - | 95.788 | None |

**rbc** (x: kind)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capillary_frame | frame | frame | 2.40/1.69/0.12 | 0.930 | 0.998 | 1.000 | 0.000 | - | 0.940 | 0.846 | 0.800 | - | - | 138.616 | None |
| capillary_average | average | average | 2.40/1.80/0.12 | 0.939 | 0.999 | 1.000 | 0.000 | - | 0.973 | 0.720 | 0.946 | - | - | 135.094 | None |

**empty** (x: kind)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| empty_average | average | average | -/-/- | 0.070 | None | None | None | None | None | None | None | None | None | 92.993 | None |
| empty_frame | frame | frame | -/-/- | 1.000 | None | None | None | None | None | None | None | None | None | 0.000 | None |

**junctions** (fitted vs truth types)

| name | fitted | truth |
|---|---|---|
| line_d2.5 | crossing, crossing |  |
| line_d3 | compound |  |
| line_d6_h0.25 | compound |  |
| line_d5_f25_h0.5 | crossing |  |
| fork_thin | pseudo-T | bifurcation |
| fork_wide | crossing, pseudo-T | bifurcation |
| fork_asym | pseudo-T | bifurcation |
| fork_acute | pseudo-T, pseudo-T | bifurcation |
| fork_thin_frame | pseudo-T | bifurcation |
| cross_a20 | compound, pseudo-T, pseudo-T | crossing |
| cross_a30 | crossing | crossing |
| cross_a45 | crossing | crossing |
| cross_a90 | crossing | crossing |
| cross_a45_dz5 | crossing | crossing |
| cross_a45_dz50 | crossing | crossing |
| cross_a45_wide_over_thin | crossing | crossing |
| cross_a45_frame | crossing | crossing |
| t_branch | pseudo-T | bifurcation |
| pseudo_t | pseudo-T | pseudo-T |
| parallel_g2 | crossing |  |
