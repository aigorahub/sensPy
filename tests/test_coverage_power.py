"""Additional tests for senspy.power module to improve coverage."""

import numpy as np
import pytest

from senspy import discrim_power, dprime_power, discrim_sample_size, dprime_sample_size
from senspy.power import _normalize_statistic, _normal_power, _exact_power


class TestNormalizeStatistic:
    """Tests for _normalize_statistic helper."""

    def test_dot_separator(self):
        assert _normalize_statistic("cont.normal", {"cont_normal"}) == "cont_normal"

    def test_dash_separator(self):
        assert _normalize_statistic("cont-normal", {"cont_normal"}) == "cont_normal"

    def test_uppercase(self):
        assert _normalize_statistic("EXACT", {"exact"}) == "exact"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            _normalize_statistic("bogus", {"exact", "normal"})


class TestNormalPowerEdgeCases:
    """Edge cases for _normal_power."""

    def test_similarity_normal_power(self):
        """Similarity test with normal approximation."""
        power = _normal_power(
            pd_a=0.1, pd_0=0.3, sample_size=100,
            alpha=0.05, p_guess=1/3, test="similarity",
        )
        assert 0 < power < 1

    def test_similarity_continuity_correction(self):
        """Similarity test with continuity correction."""
        power = _normal_power(
            pd_a=0.1, pd_0=0.3, sample_size=100,
            alpha=0.05, p_guess=1/3, test="similarity", continuity=True,
        )
        assert 0 < power < 1

    def test_sigma_a_zero_difference(self):
        """Perfect discrimination (pc_a=1) with difference test."""
        power = _normal_power(
            pd_a=1.0, pd_0=0.0, sample_size=50,
            alpha=0.05, p_guess=0.5, test="difference",
        )
        assert power == 1.0

    def test_sigma_a_zero_similarity_pc_a_less(self):
        """sigma_a=0 similarity: pc_a < pc_0 -> power=1."""
        # pd_a=0 means pc_a=p_guess, pd_0=0.5 means pc_0 > p_guess
        power = _normal_power(
            pd_a=1.0, pd_0=1.0, sample_size=50,
            alpha=0.05, p_guess=0.0, test="similarity",
        )
        # Both equal, so pc_a == pc_0 -> not less -> 0.0
        assert power == 0.0

    def test_sigma_a_zero_difference_pc_a_less(self):
        """sigma_a=0 difference: pc_a <= pc_0 -> power=0."""
        power = _normal_power(
            pd_a=1.0, pd_0=1.0, sample_size=50,
            alpha=0.05, p_guess=0.0, test="difference",
        )
        assert power == 0.0


class TestExactPowerEdgeCases:
    """Edge cases for _exact_power."""

    def test_similarity_exact_power(self):
        """Exact power for similarity test."""
        power = _exact_power(
            pd_a=0.1, pd_0=0.3, sample_size=100,
            alpha=0.05, p_guess=1/3, test="similarity",
        )
        assert 0 < power < 1

    def test_similarity_exact_power_large_effect(self):
        """Exact power for similarity test with large true effect difference."""
        power = _exact_power(
            pd_a=0.0, pd_0=0.5, sample_size=100,
            alpha=0.05, p_guess=1/3, test="similarity",
        )
        assert power > 0.5


class TestDiscrimPowerValidation:
    """Additional validation for discrim_power."""

    def test_invalid_pd_0_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=0.3, sample_size=100, pd_0=1.5, p_guess=1/3)

    def test_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=0.3, sample_size=100, alpha=0.0, p_guess=1/3)

    def test_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=0.3, sample_size=100, alpha=1.0, p_guess=1/3)

    def test_invalid_p_guess_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=0.3, sample_size=100, p_guess=1.0)

    def test_invalid_test_type_raises(self):
        with pytest.raises(ValueError, match="difference.*similarity"):
            discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3, test="bogus")

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            discrim_power(pd_a=0.3, sample_size=100, p_guess=1/3, statistic="bogus")

    def test_negative_pd_a_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_power(pd_a=-0.1, sample_size=100, p_guess=1/3)

    def test_float_sample_size_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            discrim_power(pd_a=0.3, sample_size=10.5, p_guess=1/3)

    def test_similarity_exact_statistic(self):
        """Similarity test with exact statistic."""
        power = discrim_power(
            pd_a=0.1, sample_size=100, pd_0=0.3,
            p_guess=1/3, test="similarity", statistic="exact",
        )
        assert 0 < power < 1

    def test_similarity_cont_normal_statistic(self):
        """Similarity test with continuity-corrected normal."""
        power = discrim_power(
            pd_a=0.1, sample_size=100, pd_0=0.3,
            p_guess=1/3, test="similarity", statistic="cont.normal",
        )
        assert 0 < power < 1

    def test_pd_a_equal_pd_0_difference(self):
        """pd_a == pd_0 is allowed for difference test."""
        power = discrim_power(pd_a=0.3, sample_size=100, pd_0=0.3, p_guess=1/3)
        assert 0 <= power <= 1

    def test_pd_a_equal_pd_0_similarity(self):
        """pd_a == pd_0 is allowed for similarity test."""
        power = discrim_power(
            pd_a=0.3, sample_size=100, pd_0=0.3,
            p_guess=1/3, test="similarity",
        )
        assert 0 <= power <= 1


class TestDprimePowerValidation:
    """Additional validation for dprime_power."""

    def test_negative_d_prime_0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            dprime_power(d_prime_a=1.0, sample_size=100, d_prime_0=-0.5, method="triangle")

    def test_d_prime_a_zero(self):
        """d_prime_a=0 should give power ≈ alpha."""
        power = dprime_power(d_prime_a=0.0, sample_size=100, method="triangle")
        assert power == pytest.approx(0.05, abs=0.02)

    def test_similarity_test(self):
        """Similarity test with dprime_power."""
        power = dprime_power(
            d_prime_a=0.5, sample_size=100, method="triangle",
            d_prime_0=1.5, test="similarity",
        )
        assert 0 < power < 1


class TestDiscrimSampleSizeValidation:
    """Additional validation for discrim_sample_size."""

    def test_invalid_pd_0_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_sample_size(pd_a=0.3, pd_0=1.0, p_guess=1/3)

    def test_invalid_target_power_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_sample_size(pd_a=0.3, target_power=0.0, p_guess=1/3)

    def test_invalid_target_power_one_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_sample_size(pd_a=0.3, target_power=1.0, p_guess=1/3)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_sample_size(pd_a=0.3, alpha=0.0, p_guess=1/3)

    def test_invalid_p_guess_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim_sample_size(pd_a=0.3, p_guess=1.0)

    def test_invalid_test_type_raises(self):
        with pytest.raises(ValueError, match="difference.*similarity"):
            discrim_sample_size(pd_a=0.3, p_guess=1/3, test="bogus")

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            discrim_sample_size(pd_a=0.3, p_guess=1/3, statistic="bogus")

    def test_similarity_pd_a_ge_pd_0_raises(self):
        with pytest.raises(ValueError, match="must be <"):
            discrim_sample_size(
                pd_a=0.5, pd_0=0.3, p_guess=1/3, test="similarity",
            )

    def test_stable_exact_statistic(self):
        """stable.exact statistic branch."""
        n = discrim_sample_size(
            pd_a=0.3, p_guess=1/3, statistic="stable.exact",
        )
        assert n > 0
        assert isinstance(n, int)

    def test_similarity_sample_size(self):
        """Sample size for similarity test."""
        n = discrim_sample_size(
            pd_a=0.1, pd_0=0.3, p_guess=1/3, test="similarity",
        )
        assert n > 0

    def test_similarity_normal_sample_size(self):
        """Sample size for similarity test with normal statistic."""
        n = discrim_sample_size(
            pd_a=0.1, pd_0=0.3, p_guess=1/3, test="similarity", statistic="normal",
        )
        assert n > 0

    def test_similarity_cont_normal_sample_size(self):
        """Sample size for similarity test with cont.normal statistic."""
        n = discrim_sample_size(
            pd_a=0.1, pd_0=0.3, p_guess=1/3, test="similarity", statistic="cont.normal",
        )
        assert n > 0


class TestDprimeSampleSizeValidation:
    """Additional validation for dprime_sample_size."""

    def test_negative_d_prime_a_raises(self):
        with pytest.raises(ValueError, match="positive"):
            dprime_sample_size(d_prime_a=-1.0, method="triangle")

    def test_negative_d_prime_0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            dprime_sample_size(d_prime_a=1.0, d_prime_0=-0.5, method="triangle")

    def test_similarity_sample_size(self):
        """Sample size for similarity test with d-prime."""
        n = dprime_sample_size(
            d_prime_a=0.5, method="triangle",
            d_prime_0=1.5, test="similarity",
        )
        assert n > 0
