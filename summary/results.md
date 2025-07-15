- Got similar result when used minmax to when I used standarization
- R2 = 0.5702 -> validation (v1)


We notice that removing outliers causes the performance to decrease instead of imporving. if we use the quantile method,
Due to the high volume of outliers our data gets so small which affects the performance directly, so it is better to decrease the outlier
range to make sure we only remove extreme outliers.


slope is useless

distance per passenger is also useless.

MAIN lesson:
1. Don't think about if the feature is useful or not, just create it anyways, it is hard to know what can be useful and what's not

(WE have DAta leakage in calculating the IQR of validation set, we should use train IQR)