# Seasonal Baseline

The reference forecast is the mean sale for the matching weekday over the eight weeks ending at each forecast origin. It predicts all 16 horizons without reading sales from inside the target block.

Across the three declared development origins, mean RMSLE is **0.501865**. These folds are used for model development; the separate 2017-07-30 final holdout is not used here.

