# cM_6 Dataset: Where Did the 302 Pairs Go?

| Source | Count |
|--------|-------|
| Kinship data (raw) | 3,786 |
| Pairs with statistics | 3,507 |
| **Final merged dataset** | **3,484** |
| Training set (80%) | 2,787 |
| Validation set (20%) | 697 |

**The math:** 2,787 + 697 = **3,484** ✓

---

## The Gaps Explained

### Gap 1: Kinship → Final (3,786 → 3,484 = 302 pairs lost)

**302 pairs had kinship labels but lacked statistical records.**

These pairs existed in the kinship file but were never analyzed for distributional statistics. They had IBD metrics but no percentiles, cM thresholds, or segment aggregates to merge with.

### Gap 2: Statistics → Final (3,507 → 3,484 = 23 pairs lost)

**23 pairs had statistics but lacked kinship labels.**

These pairs were statistically analyzed but never assigned a kinship classification. They exist in the merged_info.out files but have no corresponding record in the kinship file.

### Why 3,507 and 3,484?

During the merge:
- Kinship data: **3,786 pairs** (no statistics for 302)
- Statistics data: **3,507 pairs** (no kinship labels for 23)
- Final overlap: **3,484 pairs** (both kinship AND statistics)

### Examples of Dropped Pairs (all labeled UN):

1. `1-1_vs_30-1` — Had IBD data, lacked statistics
2. `1-1_vs_37-1` — Had IBD data, lacked statistics
3. `1-1_vs_46-1` — Had IBD data, lacked statistics
4. `1-2_vs_30-1` — Had IBD data, lacked statistics
5. `1-2_vs_31-1` — Had IBD data, lacked statistics
... (297 more UN-labeled pairs)

### All 302 Dropped Pairs Share One Feature:
- **Kinship label:** UN (unclassified/ambiguous)
- **Missing data:** Distributional statistics (percentiles, cM thresholds, segment aggregates)
- **Impact:** These pairs were excluded to ensure data quality for model training
