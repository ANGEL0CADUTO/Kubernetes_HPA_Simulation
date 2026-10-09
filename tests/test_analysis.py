import tempfile
import unittest
from pathlib import Path

from src import config
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.acs import batch_means, compute_batch_size


class AnalysisTests(unittest.TestCase):
    def test_constant_observations_have_zero_width_confidence_interval(self):
        values = [3.0] * 512
        b, k, rho = compute_batch_size(values, 256, 0.2)
        result = batch_means(values, b, k)
        self.assertEqual(rho, 0)
        self.assertEqual(result['mean'], 3)
        self.assertEqual(result['ci'], (3, 3))

    def test_insufficient_batches_do_not_produce_an_estimate(self):
        self.assertEqual(compute_batch_size([1.0] * 64, 256, 0.2), (None, None, None))

    def test_report_handles_an_unavailable_confidence_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            SteadyStatePlotter(config).generate_final_report(
                all_steady_values={'WFQ': [1.0, 2.0]},
                all_results={'WFQ': None},
                output_dir=directory,
            )
            self.assertTrue((Path(directory) / 'response_time_ci_wfq.png').is_file())
            self.assertFalse(list(Path(directory).glob('cumulative*')))


if __name__ == '__main__':
    unittest.main()
