# Feature Importance

Importances are aggregated across the five hash-verified LightGBM folds. Gain is descriptive and does not establish causal value; ablation is required before removing a feature.

- Artifact mode: `full` training / `full` test
- Feature schema: `3`
- Candidate sampling schema: `2`

| Rank | Feature | Gain share | Gain mean | Gain std | Split mean | Nonzero folds |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `tfidf_cosine_delta_mean` | 46.17% | 3898086.469 | 7012.963 | 8931.6 | 5 |
| 2 | `candidate_count_log1p` | 8.45% | 713283.121 | 11959.574 | 2152.6 | 5 |
| 3 | `query_title_overlap_rank` | 6.95% | 586932.951 | 6793.418 | 8123.8 | 5 |
| 4 | `tfidf_cosine_rank` | 5.32% | 449171.666 | 6819.850 | 8276.0 | 5 |
| 5 | `tfidf_cosine` | 4.12% | 348180.574 | 7510.657 | 10028.4 | 5 |
| 6 | `query_category_overlap_rank` | 3.19% | 269081.711 | 9026.079 | 12609.6 | 5 |
| 7 | `query_brand_match` | 3.08% | 259865.604 | 9671.728 | 1632.4 | 5 |
| 8 | `query_title_coverage` | 2.86% | 241254.439 | 4811.168 | 2281.0 | 5 |
| 9 | `query_title_coverage_rank` | 2.62% | 220813.519 | 4327.346 | 10210.4 | 5 |
| 10 | `tfidf_cosine_gap` | 1.73% | 146068.148 | 7025.481 | 10543.4 | 5 |
| 11 | `query_category_coverage` | 1.49% | 126193.513 | 6147.660 | 2220.0 | 5 |
| 12 | `query_title_overlap_gap` | 1.36% | 114528.311 | 5434.831 | 8502.0 | 5 |
| 13 | `query_title_overlap_delta_mean` | 1.33% | 112147.070 | 4701.911 | 8075.0 | 5 |
| 14 | `cat_depth` | 1.23% | 104061.035 | 3200.572 | 3619.6 | 5 |
| 15 | `query_len` | 1.22% | 103091.103 | 4951.762 | 8302.0 | 5 |
| 16 | `title_len` | 1.20% | 101716.072 | 6348.153 | 7864.4 | 5 |
| 17 | `query_category_overlap` | 1.01% | 85556.295 | 3701.968 | 4739.0 | 5 |
| 18 | `query_token_count` | 1.01% | 85537.036 | 3327.747 | 1885.8 | 5 |
| 19 | `title_token_count` | 0.83% | 70455.795 | 2153.857 | 3592.2 | 5 |
| 20 | `gender_match` | 0.80% | 67367.609 | 931.903 | 843.6 | 5 |
| 21 | `query_cat_l3_overlap` | 0.73% | 61774.800 | 3085.189 | 3148.0 | 5 |
| 22 | `query_cat_l2_overlap` | 0.61% | 51758.876 | 1568.303 | 2042.4 | 5 |
| 23 | `demographic_conflict` | 0.59% | 49733.186 | 1524.350 | 285.0 | 5 |
| 24 | `query_color_match` | 0.49% | 40952.008 | 1130.787 | 705.0 | 5 |
| 25 | `age_group_match` | 0.44% | 36957.032 | 1128.679 | 631.0 | 5 |
| 26 | `query_title_overlap` | 0.39% | 33059.035 | 2120.427 | 2486.8 | 5 |
| 27 | `query_cat_l1_overlap` | 0.26% | 22046.424 | 1542.821 | 988.0 | 5 |
| 28 | `query_title_precision` | 0.22% | 18587.748 | 1474.375 | 1950.0 | 5 |
| 29 | `query_title_phrase` | 0.18% | 15567.261 | 875.242 | 410.4 | 5 |
| 30 | `query_model_token_conflict` | 0.05% | 4579.899 | 547.386 | 150.2 | 5 |
| 31 | `query_material_match` | 0.03% | 2511.961 | 208.394 | 195.2 | 5 |
| 32 | `query_model_token_match` | 0.03% | 2418.271 | 241.558 | 190.4 | 5 |
| 33 | `query_size_match` | 0.00% | 0.000 | 0.000 | 0.0 | 0 |

Features with zero or unstable gain remain in the contract until a grouped, cross-fitted ablation demonstrates a non-negative removal decision.
