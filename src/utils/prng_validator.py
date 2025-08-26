import numpy as np
from scipy.stats import stats, chi2, pearsonr


class PRNGValidator:
    """
    Implementa test per validare la qualità del PRNG secondo Kurkowski.
    """

    @staticmethod
    def test_uniformity(rng, num_samples=10000):
        """Test chi-square per uniformità."""
        samples = [rng.random() for _ in range(num_samples)]

        # Divide in 10 bins
        counts, _ = np.histogram(samples, bins=10, range=(0, 1))
        expected = num_samples / 10

        chi_square = sum((count - expected)**2 / expected for count in counts)
        p_value = 1 - chi2.cdf(chi_square, df=9)

        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'uniform': p_value > 0.05,
            'test': 'Chi-square uniformity'
        }

    @staticmethod
    def test_independence(rng, num_pairs=10000):
        """Test di indipendenza tra coppie consecutive."""
        pairs = [(rng.random(), rng.random()) for _ in range(num_pairs)]
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        correlation, p_value = pearsonr(x_vals, y_vals)

        return {
            'correlation': correlation,
            'p_value': p_value,
            'independent': abs(correlation) < 0.05 < p_value,
            'test': 'Pearson correlation'
        }

    @staticmethod
    def validate_lehmer_for_simulation(lehmer_rng, num_dimensions=3, num_samples=50000):
        """
        Valida se il Lehmer RNG è adeguato per simulazione multidimensionale.
        Basato sui criteri in Kurkowski.
        """
        print("\n--- VALIDAZIONE PRNG LEHMER ---")

        # Genera stream per diverse dimensioni
        seeds = lehmer_rng.get_numpy_seeds(count=num_dimensions)
        rngs = [np.random.default_rng(seed) for seed in seeds]

        results = {}

        # Test ogni dimensione
        for i, rng in enumerate(rngs):
            uniformity = PRNGValidator.test_uniformity(rng, num_samples)
            independence = PRNGValidator.test_independence(rng, num_samples//2)

            results[f'stream_{i}'] = {
                'uniformity': uniformity,
                'independence': independence
            }

            print(f"Stream {i}: Uniform={uniformity['uniform']}, Independent={independence['independent']}")

        # Test correlazione cross-stream
        samples_per_stream = [[rngs[i].random() for _ in range(1000)] for i in range(num_dimensions)]

        print("\nCorrelazioni Cross-Stream:")
        cross_correlations_valid = True
        for i in range(num_dimensions):
            for j in range(i+1, num_dimensions):
                corr, p_val = stats.pearsonr(samples_per_stream[i], samples_per_stream[j])
                valid = abs(corr) < 0.05
                cross_correlations_valid &= valid
                print(f"Stream {i} vs {j}: r={corr:.4f}, p={p_val:.4f}, Valid={valid}")

        overall_valid = all(r['uniformity']['uniform'] and r['independence']['independent']
                            for r in results.values()) and cross_correlations_valid

        print(f"\nVALIDITÀ COMPLESSIVA PRNG: {'OK' if overall_valid else 'FALLITA'}")
        return overall_valid, results