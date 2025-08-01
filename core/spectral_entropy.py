
"""
SpectralEntropy — Novel hallucination detection metric.

Motivation:
    EigenScore uses the log-determinant (sum of log eigenvalues) as a
    divergence measure. However, it is sensitive to the *magnitude* of
    eigenvalues and can be dominated by a single large eigenvalue.

    SpectralEntropy instead normalizes eigenvalues into a probability
    distribution and computes Shannon entropy over them.  This captures
    *how uniformly* the variance is spread across the embedding dimensions:

    - Low SpectralEntropy  → one dominant eigenvalue → responses are
      semantically clustered → model is confident → less hallucination.
    - High SpectralEntropy → eigenvalues uniformly distributed → responses
      span many directions → model is uncertain → likely hallucination.

    This is complementary to EigenScore:
    EigenScore measures total semantic spread (volume), while
    SpectralEntropy measures the uniformity of that spread (shape).
"""