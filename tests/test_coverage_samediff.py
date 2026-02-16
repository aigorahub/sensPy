"""Additional tests for senspy.samediff module to improve coverage."""

import numpy as np
import pytest

from senspy import samediff


class TestSameDiffAdditionalCases:
    """Additional boundary/edge case tests."""

    def test_case_112_no_same_sample_data(self):
        """Case 1.12: ss=0, ds=0 -> optimize over both."""
        result = samediff(0, 0, 4, 9)
        assert result.case == 1.12
        assert np.isfinite(result.log_likelihood)

    def test_case_122_ds_zero_only(self):
        """Case 1.22: ds=0 (but ss>0, sd>0, dd>0) -> tau=Inf, delta=Inf."""
        result = samediff(8, 0, 4, 9)
        assert result.case == 1.22
        assert result.tau == np.inf
        assert result.delta == np.inf
        assert np.isfinite(result.log_likelihood)

    def test_case_13_no_diff_sample_data(self):
        """Case 1.3: sd=0, dd=0 -> optimize tau, delta=NA."""
        result = samediff(8, 5, 0, 0)
        assert result.case == 1.3
        assert np.isnan(result.delta)
        assert np.isfinite(result.tau)
        assert result.tau > 0

    def test_case_13_se_available(self):
        """Case 1.3 with vcov: se_tau should be computed."""
        result = samediff(8, 5, 0, 0, vcov=True)
        assert result.case == 1.3
        # se_tau may be available if Hessian is negative
        if result.se_tau is not None:
            assert result.se_tau > 0

    def test_case_2_dd_zero(self):
        """Case 2: dd=0 -> delta=0."""
        result = samediff(8, 5, 4, 0)
        assert result.case == 2.0
        assert result.delta == 0.0
        assert np.isfinite(result.tau)

    def test_case_2_ss_zero_with_deltas(self):
        """Case 2: ss=0 with is_delta false."""
        result = samediff(0, 5, 4, 9)
        # ss=0 -> is_delta check: ss/(ds+ss) = 0/5 < sd/(sd+dd) = 4/13
        # -> is_delta = False -> case 2
        assert result.case == 2.0
        assert result.delta == 0.0

    def test_case_3_sd_zero_vcov(self):
        """Case 3: sd=0 with vcov computation."""
        result = samediff(8, 5, 0, 9, vcov=True)
        assert result.case == 3.0
        assert result.delta == np.inf
        if result.se_tau is not None:
            assert result.se_tau > 0

    def test_vcov_false_all_cases(self):
        """Test vcov=False across various cases."""
        # General case
        result = samediff(8, 5, 4, 9, vcov=False)
        assert result.vcov is None

        # Case 1.3
        result = samediff(8, 5, 0, 0, vcov=False)
        assert result.vcov is None

        # Case 3
        result = samediff(8, 5, 0, 9, vcov=False)
        assert result.vcov is None

    def test_large_delta_root_finding(self):
        """Test with data that requires extended root-finding range."""
        # Very high discrimination: P(same|same) >> P(same|diff)
        result = samediff(95, 5, 2, 98)
        assert np.isfinite(result.tau)
        assert np.isfinite(result.delta)
        assert result.delta > 0

    def test_str_with_inf_tau(self):
        """Test __str__ with infinite tau."""
        result = samediff(8, 0, 4, 0)
        output = str(result)
        assert "Inf" in output

    def test_str_with_nan_delta(self):
        """Test __str__ with NaN delta."""
        result = samediff(0, 5, 0, 9)
        output = str(result)
        assert "NA" in output

    def test_str_with_inf_delta(self):
        """Test __str__ with Inf delta."""
        result = samediff(8, 0, 0, 9)
        output = str(result)
        assert "Inf" in output

    def test_general_case_fisher_info(self):
        """Test general case produces valid Fisher information."""
        result = samediff(45, 5, 20, 30)
        assert result.vcov is not None
        assert np.all(np.isfinite(result.vcov))
        # Variance-covariance should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(result.vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_convergence_stored(self):
        """Cases with optimization store convergence."""
        result = samediff(8, 5, 0, 0)  # Case 1.3 with optimization
        assert result.convergence is not None

    def test_data_stored_as_int(self):
        """Data is stored as int array."""
        result = samediff(8, 5, 4, 9)
        assert result.data.dtype == int or np.issubdtype(result.data.dtype, np.integer)
