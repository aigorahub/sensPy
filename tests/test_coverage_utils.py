"""Additional tests for senspy.utils module to improve coverage."""

import numpy as np
import pytest

from senspy.utils.stats import delimit, normal_pvalue, find_critical
from senspy.utils.stats import test_critical as _test_critical
from senspy.utils.transforms import pc_to_pd, pd_to_pc, rescale


class TestDelimit:
    """Tests for delimit function."""

    def test_both_bounds(self):
        result = delimit([0.1, 0.5, 0.9], lower=0.2, upper=0.8)
        np.testing.assert_allclose(result, [0.2, 0.5, 0.8])

    def test_lower_only(self):
        result = delimit([-1.0, 0.5, 2.0], lower=0.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 2.0])

    def test_upper_only(self):
        result = delimit([-1.0, 0.5, 2.0], upper=1.0)
        np.testing.assert_allclose(result, [-1.0, 0.5, 1.0])

    def test_no_bounds(self):
        """Both None -> returns unchanged copy."""
        result = delimit([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_scalar_input(self):
        result = delimit(-0.5, lower=0.0)
        assert result[0] == 0.0

    def test_lower_ge_upper_raises(self):
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            delimit([0.5], lower=1.0, upper=0.5)

    def test_lower_eq_upper_raises(self):
        with pytest.raises(ValueError, match="lower.*must be less than upper"):
            delimit([0.5], lower=0.5, upper=0.5)


class TestNormalPvalue:
    """Tests for normal_pvalue function."""

    def test_two_sided(self):
        p = normal_pvalue(1.96, alternative="two.sided")
        assert p[0] == pytest.approx(0.05, abs=0.001)

    def test_greater(self):
        p = normal_pvalue(1.645, alternative="greater")
        assert p[0] == pytest.approx(0.05, abs=0.001)

    def test_less(self):
        p = normal_pvalue(-1.645, alternative="less")
        assert p[0] == pytest.approx(0.05, abs=0.001)

    def test_less_positive_stat(self):
        """Less alternative with positive stat -> large p-value."""
        p = normal_pvalue(2.0, alternative="less")
        assert p[0] > 0.5

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="Unknown alternative"):
            normal_pvalue(1.96, alternative="bogus")

    def test_array_input(self):
        p = normal_pvalue([1.96, -1.96], alternative="two.sided")
        assert len(p) == 2
        np.testing.assert_allclose(p[0], p[1])

    def test_zero_stat_two_sided(self):
        p = normal_pvalue(0.0, alternative="two.sided")
        assert p[0] == pytest.approx(1.0)

    def test_zero_stat_greater(self):
        p = normal_pvalue(0.0, alternative="greater")
        assert p[0] == pytest.approx(0.5)


class TestFindCritical:
    """Tests for find_critical function."""

    def test_difference_basic(self):
        xcr = find_critical(sample_size=100, alpha=0.05, p0=0.5)
        assert isinstance(xcr, int)
        assert xcr > 50

    def test_similarity_basic(self):
        xcr = find_critical(sample_size=100, alpha=0.05, p0=0.5, test="similarity")
        assert isinstance(xcr, int)
        assert xcr < 50

    def test_difference_with_pd0(self):
        xcr = find_critical(sample_size=100, alpha=0.05, p0=1/3, pd0=0.2)
        assert isinstance(xcr, int)

    def test_similarity_with_pd0(self):
        xcr = find_critical(sample_size=100, alpha=0.05, p0=1/3, pd0=0.5, test="similarity")
        assert isinstance(xcr, int)

    def test_invalid_sample_size_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            find_critical(sample_size=0)

    def test_float_sample_size_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            find_critical(sample_size=10.5)

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be between"):
            find_critical(sample_size=100, alpha=0.0)

    def test_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha must be between"):
            find_critical(sample_size=100, alpha=1.0)

    def test_invalid_p0_zero_raises(self):
        with pytest.raises(ValueError, match="p0 must be between"):
            find_critical(sample_size=100, p0=0.0)

    def test_invalid_pd0_raises(self):
        with pytest.raises(ValueError, match="pd0 must be between"):
            find_critical(sample_size=100, pd0=-0.1)

    def test_invalid_test_raises(self):
        with pytest.raises(ValueError, match="Unknown test"):
            find_critical(sample_size=100, test="bogus")

    def test_small_sample_difference(self):
        xcr = find_critical(sample_size=10, alpha=0.05, p0=1/3)
        assert isinstance(xcr, int)

    def test_small_sample_similarity(self):
        xcr = find_critical(sample_size=10, alpha=0.05, p0=0.5, pd0=0.5, test="similarity")
        assert isinstance(xcr, int)


class TestTestCritical:
    """Tests for test_critical function (previously untested)."""

    def test_difference_valid_critical(self):
        """test_critical should confirm the value from find_critical."""
        xcr = find_critical(sample_size=100, alpha=0.05, p0=0.5)
        assert _test_critical(xcr, sample_size=100, p_correct=0.5, alpha=0.05, test="difference")

    def test_difference_wrong_value(self):
        """A value that is not the critical value should return False."""
        xcr = find_critical(sample_size=100, alpha=0.05, p0=0.5)
        assert not _test_critical(xcr + 5, sample_size=100, p_correct=0.5, alpha=0.05, test="difference")

    def test_similarity_valid_critical(self):
        """test_critical should confirm similarity critical value."""
        xcr = find_critical(sample_size=100, alpha=0.05, p0=1/3, pd0=0.3, test="similarity")
        pc = 0.3 + (1/3) * (1 - 0.3)
        assert _test_critical(xcr, sample_size=100, p_correct=pc, alpha=0.05, test="similarity")

    def test_greater_alias(self):
        """'greater' should work same as 'difference'."""
        xcr = find_critical(sample_size=50, alpha=0.05, p0=0.5)
        assert _test_critical(xcr, sample_size=50, p_correct=0.5, alpha=0.05, test="greater")

    def test_less_alias(self):
        """'less' should work same as 'similarity'."""
        xcr = find_critical(sample_size=50, alpha=0.05, p0=1/3, pd0=0.3, test="similarity")
        pc = 0.3 + (1/3) * (1 - 0.3)
        assert _test_critical(xcr, sample_size=50, p_correct=pc, alpha=0.05, test="less")

    def test_invalid_test_raises(self):
        with pytest.raises(ValueError, match="Unknown test"):
            _test_critical(50, sample_size=100, test="bogus")


class TestPcToPd:
    """Tests for pc_to_pd function."""

    def test_basic(self):
        result = pc_to_pd(0.8, p_guess=0.5)
        assert result[0] == pytest.approx(0.6)

    def test_at_chance(self):
        result = pc_to_pd(0.5, p_guess=0.5)
        assert result[0] == pytest.approx(0.0)

    def test_below_chance_clamped(self):
        """pc below p_guess should give pd = 0."""
        result = pc_to_pd(0.2, p_guess=0.5)
        assert result[0] == 0.0

    def test_perfect(self):
        result = pc_to_pd(1.0, p_guess=1/3)
        assert result[0] == pytest.approx(1.0)

    def test_invalid_p_guess_raises(self):
        with pytest.raises(ValueError, match="p_guess must be in"):
            pc_to_pd(0.5, p_guess=-0.1)

    def test_array_input(self):
        result = pc_to_pd([0.5, 0.7, 1.0], p_guess=0.5)
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.0, 0.4, 1.0])

    def test_p_guess_zero(self):
        result = pc_to_pd(0.5, p_guess=0.0)
        assert result[0] == pytest.approx(0.5)


class TestPdToPc:
    """Tests for pd_to_pc function."""

    def test_basic(self):
        result = pd_to_pc(0.6, p_guess=0.5)
        assert result[0] == pytest.approx(0.8)

    def test_zero_pd(self):
        result = pd_to_pc(0.0, p_guess=1/3)
        assert result[0] == pytest.approx(1/3)

    def test_one_pd(self):
        result = pd_to_pc(1.0, p_guess=1/3)
        assert result[0] == pytest.approx(1.0)

    def test_invalid_p_guess_raises(self):
        with pytest.raises(ValueError, match="p_guess must be in"):
            pd_to_pc(0.5, p_guess=1.5)


class TestRescale:
    """Tests for rescale function."""

    def test_from_d_prime(self):
        result = rescale(d_prime=1.5, method="triangle")
        assert np.isfinite(result.pc)
        assert np.isfinite(result.pd)
        assert np.isfinite(result.d_prime)
        assert result.d_prime == pytest.approx(1.5)

    def test_from_pc(self):
        result = rescale(pc=0.8, method="twoafc")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.pd)
        assert result.pc == pytest.approx(0.8)

    def test_from_pd(self):
        result = rescale(pd=0.5, method="triangle")
        assert np.isfinite(result.pc)
        assert np.isfinite(result.d_prime)
        assert result.pd == pytest.approx(0.5)

    def test_from_d_prime_with_se(self):
        result = rescale(d_prime=1.5, se=0.2, method="triangle")
        assert result.se_pc is not None
        assert result.se_pd is not None
        assert result.se_d_prime is not None
        assert result.se_d_prime == pytest.approx(0.2)

    def test_from_pc_with_se(self):
        result = rescale(pc=0.8, se=0.05, method="twoafc")
        assert result.se_d_prime is not None
        assert result.se_pd is not None

    def test_from_pd_with_se(self):
        result = rescale(pd=0.5, se=0.1, method="triangle")
        assert result.se_pc is not None
        assert result.se_d_prime is not None

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            rescale(method="triangle")

    def test_multiple_inputs_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            rescale(pc=0.8, d_prime=1.5, method="triangle")

    def test_pc_below_p_guess(self):
        """pc below chance should be clamped."""
        result = rescale(pc=0.2, method="triangle")
        assert result.d_prime == pytest.approx(0.0)
        assert result.pd == pytest.approx(0.0)

    def test_pc_below_p_guess_with_se(self):
        """When pc < p_guess, SE should be NaN."""
        result = rescale(pc=0.2, se=0.05, method="triangle")
        assert np.isnan(result.se_pc)

    def test_negative_d_prime_clamped(self):
        """Negative d_prime should be clamped to 0."""
        result = rescale(d_prime=-0.5, method="triangle")
        assert result.d_prime == 0.0

    def test_array_input_d_prime(self):
        result = rescale(d_prime=[0.5, 1.0, 2.0], method="twoafc")
        assert len(result.pc) == 3
        assert len(result.pd) == 3

    def test_roundtrip_pc_d_prime(self):
        """Converting d_prime->pc->d_prime should be consistent."""
        r1 = rescale(d_prime=1.5, method="triangle")
        r2 = rescale(pc=r1.pc, method="triangle")
        assert r2.d_prime == pytest.approx(1.5, rel=1e-3)
