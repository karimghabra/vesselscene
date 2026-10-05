## probes: nothing 

probe_score 0.1422  digest 70600d88a3f17f13  1 s  (34 probes)

| sweep | mean | threshold |
|---|---|---|
| width | 0.167 | None (smallest diameter detected) |
| blur | 0.167 | None (largest defocus detected with width err < 0.25) |
| contrast | 0.167 | None (smallest haematocrit detected) |
| corner | 0.167 | 0/1 (thin, defocused, low contrast detected) |
| fork | 0.000 | 0/5 (forks matched and typed 3-way) |
| crossing_angle | 0.000 | None (smallest angle typed a crossing) |
| crossing_depth | 0.000 | None (smallest depth gap typed a crossing) |
| crossing_width | 0.000 | 0/1 (unequal crossing typed a crossing) |
| t | 0.000 | 0/2 (T junctions matched and typed 3-way) |
| gap | 0.167 | None (smallest wall gap resolved (2 lines)) |
| end | 0.167 | 0/2 (ends traced without a false junction) |
| rbc | 0.167 | 0/2 (capillary detected) |
| empty | 1.000 | 2/2 empty; max len 0 px (nothing fitted) |
| gap (truth) | | 2.0 (smallest gap the observable truth resolves) |

**width** (x: diameter px)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d2.5 | 2.500 | average | 1.25/1.80/0.06 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d3 | 3.000 | average | 1.50/1.80/0.08 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d6 | 6.000 | average | 3.00/1.80/0.15 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d12 | 12.000 | average | 6.00/1.80/0.29 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**blur** (x: focus offset d_c)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d6 | 0.000 | average | 3.00/1.80/0.15 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d6_f25 | 25.000 | average | 3.00/2.34/0.15 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d6_f50 | 50.000 | average | 3.00/3.50/0.15 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**contrast** (x: haematocrit)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d6 | 1.000 | average | 3.00/1.80/0.15 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d6_h0.5 | 0.500 | average | 3.00/1.80/0.08 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| line_d6_h0.25 | 0.250 | average | 3.00/1.80/0.04 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**corner** (x: -)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line_d5_f25_h0.5 | d5 f25 h0.5 | average | 2.50/2.34/0.07 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**fork** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fork_thin | thin r0=2 | average | 1.59/1.80/0.08 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| fork_wide | wide r0=6 | average | 4.76/1.80/0.24 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| fork_asym | asym 6>5.5+3.7 | average | 5.50/1.80/0.27 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| fork_acute | acute 2x15deg | average | 3.17/1.80/0.16 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| fork_thin_frame | thin frame | frame | 1.59/1.69/0.08 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |

**crossing_angle** (x: angle deg)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a20 | 20 | average | 3.00/2.21/0.13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| cross_a30 | 30 | average | 3.00/2.21/0.13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| cross_a45 | 45 | average | 3.00/2.21/0.13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| cross_a90 | 90 | average | 3.00/2.00/0.14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| cross_a45_frame | 45 frame | frame | 3.00/2.12/0.13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |

**crossing_depth** (x: depth gap d_c)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a45_dz5 | 5.000 | average | 3.00/1.83/0.15 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| cross_a45_dz50 | 50.000 | average | 3.00/3.61/0.09 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |

**crossing_width** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_a45_wide_over_thin | 6 over 1.5 | average | 1.50/2.21/0.06 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |

**t** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t_branch | T branch | average | 4.95/1.80/0.25 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |
| pseudo_t | pseudo-T | average | 5.00/1.80/0.25 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 0.000 | None |

**gap** (x: wall gap px)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parallel_g1 | 1.000 | average | 2.00/1.80/0.11 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | 0 |
| parallel_g2 | 2.000 | average | 2.00/1.80/0.11 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | 0 |
| parallel_g3 | 3.000 | average | 2.00/1.80/0.11 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | 0 |
| parallel_g5 | 5.000 | average | 2.00/1.80/0.11 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | 0 |

**end** (x: geometry)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| end_dive | dive | average | 2.50/2.43/0.14 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| end_blind | blind end | average | 2.00/1.80/0.10 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**rbc** (x: kind)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capillary_frame | frame | frame | 2.40/1.69/0.12 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |
| capillary_average | average | average | 2.40/1.80/0.12 | 0.167 | 0.333 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | - | - | 0.000 | None |

**empty** (x: kind)

| name | x | kind | r/s/a true | probe_composite | graph_probe | centreline_recall | junction_f1_strict | crossing_recall | pos | width | explained | explained_junction | junction_bias | fitted_len_px | n_lines_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| empty_average | average | average | -/-/- | 1.000 | None | None | None | None | None | None | None | None | None | 0.000 | None |
| empty_frame | frame | frame | -/-/- | 1.000 | None | None | None | None | None | None | None | None | None | 0.000 | None |

**junctions** (fitted vs truth types)

| name | fitted | truth |
|---|---|---|
| fork_thin |  | bifurcation |
| fork_wide |  | bifurcation |
| fork_asym |  | bifurcation |
| fork_acute |  | bifurcation |
| fork_thin_frame |  | bifurcation |
| cross_a20 |  | crossing |
| cross_a30 |  | crossing |
| cross_a45 |  | crossing |
| cross_a90 |  | crossing |
| cross_a45_dz5 |  | crossing |
| cross_a45_dz50 |  | crossing |
| cross_a45_wide_over_thin |  | crossing |
| cross_a45_frame |  | crossing |
| t_branch |  | bifurcation |
| pseudo_t |  | pseudo-T |
