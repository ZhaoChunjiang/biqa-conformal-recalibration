# Statistical notes

## Interval-score tests against the reference value 1

The paper's primary evidence is descriptive and operating-regime based:
- direction-level medians and IQRs;
- seed-level rates with interval score below 1;
- the pre-specified operational criterion combining coverage error, clipped width, and interval score.

A one-sided Wilcoxon signed-rank test against interval score 1 is retained as an auxiliary magnitude-sensitive statistic. The signed-rank test relies on a symmetry assumption for the nonzero paired differences under the null.

To reduce dependence on that assumption, the code also reports an **exact sign/binomial test**:
- ties at exactly 1 are discarded;
- only the sign of each seed-level difference `IS - 1` is used;
- the null probability of either sign is 0.5.

The sign test is less powerful because it ignores magnitudes, but it does not require symmetry of the difference distribution. Statistical p-values are therefore treated as supporting diagnostics, not as the definition of an informative operating regime.
