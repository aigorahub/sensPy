"""Additional tests for senspy.discrim module to improve coverage."""

import numpy as np
import pytest

from senspy import discrim
from senspy.core.types import Statistic


class TestDiscrimStatistics:
    """Test different statistic types."""

    def test_likelihood_statistic(self):
        result = discrim(80, 100, method="triangle", statistic="likelihood")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)
        ci = result.confint()
        assert ci[0] < result.d_prime < ci[1]

    def test_wald_statistic(self):
        result = discrim(80, 100, method="triangle", statistic="wald")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)
        ci = result.confint()
        assert ci[0] < result.d_prime < ci[1]

    def test_score_statistic(self):
        result = discrim(80, 100, method="triangle", statistic="score")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)
        ci = result.confint()
        assert ci[0] < result.d_prime < ci[1]

    def test_statistic_enum(self):
        """Test passing Statistic enum directly."""
        result = discrim(80, 100, method="triangle", statistic=Statistic.LIKELIHOOD)
        assert np.isfinite(result.d_prime)

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            discrim(80, 100, method="triangle", statistic="bogus")


class TestDiscrimSimilarityTests:
    """Tests for similarity test mode."""

    def test_similarity_with_d_prime0(self):
        result = discrim(
            40, 100, method="triangle",
            d_prime0=1.0, test="similarity",
        )
        assert np.isfinite(result.p_value)

    def test_similarity_with_pd0(self):
        result = discrim(
            40, 100, method="triangle",
            pd0=0.3, test="similarity",
        )
        assert np.isfinite(result.p_value)

    def test_similarity_requires_null(self):
        """Similarity test requires d_prime0 or pd0."""
        with pytest.raises(ValueError, match="must be specified"):
            discrim(40, 100, method="triangle", test="similarity")

    def test_both_d_prime0_and_pd0_raises(self):
        with pytest.raises(ValueError, match="Only specify one"):
            discrim(40, 100, method="triangle", d_prime0=1.0, pd0=0.3)

    def test_negative_d_prime0_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            discrim(40, 100, method="triangle", d_prime0=-0.5)

    def test_invalid_pd0_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim(40, 100, method="triangle", pd0=1.5)

    def test_similarity_with_likelihood(self):
        result = discrim(
            40, 100, method="triangle",
            d_prime0=1.0, test="similarity", statistic="likelihood",
        )
        assert np.isfinite(result.p_value)

    def test_similarity_with_wald(self):
        result = discrim(
            40, 100, method="triangle",
            d_prime0=1.0, test="similarity", statistic="wald",
        )
        assert np.isfinite(result.p_value)

    def test_similarity_with_score(self):
        result = discrim(
            40, 100, method="triangle",
            d_prime0=1.0, test="similarity", statistic="score",
        )
        assert np.isfinite(result.p_value)


class TestDiscrimBoundaryCases:
    """Boundary case tests."""

    def test_all_correct(self):
        """All correct: pc=1, d_prime=Inf or large."""
        result = discrim(100, 100, method="triangle")
        assert result.pc == 1.0
        assert np.isnan(result.se_d_prime) or result.se_d_prime == 0.0
        ci = result.confint()
        assert ci[1] == np.inf or ci[1] > 10

    def test_at_chance(self):
        """Correct == expected by chance: d_prime=0."""
        result = discrim(33, 100, method="triangle")
        assert result.d_prime == 0.0
        assert result.pd == 0.0

    def test_below_chance(self):
        """Below chance level: d_prime should be 0."""
        result = discrim(20, 100, method="triangle")
        assert result.d_prime == 0.0
        assert result.pd == 0.0

    def test_zero_correct(self):
        """Zero correct responses."""
        result = discrim(0, 100, method="triangle")
        assert result.d_prime == 0.0
        assert result.pc == 0.0

    def test_wald_boundary_pc_zero(self):
        """Wald stat when pc=0 (se_pc=0)."""
        result = discrim(0, 100, method="twoafc", statistic="wald")
        assert np.isfinite(result.p_value) or result.p_value == 1.0

    def test_wald_boundary_pc_one(self):
        """Wald stat when pc=1 (se_pc=0) and pc > pc0."""
        result = discrim(100, 100, method="twoafc", statistic="wald")
        assert result.p_value == pytest.approx(0.0, abs=1e-10)

    def test_ci_lower_below_pguess(self):
        """CI where lower bound is below p_guess."""
        result = discrim(35, 100, method="triangle")
        ci = result.confint(parameter="d_prime")
        assert ci[0] == 0.0  # Clamped at 0

    def test_confint_pc(self):
        result = discrim(80, 100, method="triangle")
        ci = result.confint(parameter="pc")
        assert ci[0] < result.pc < ci[1]

    def test_confint_pd(self):
        result = discrim(80, 100, method="triangle")
        ci = result.confint(parameter="pd")
        assert ci[0] < result.pd < ci[1]


class TestDiscrimInputValidation:
    """Input validation tests."""

    def test_float_correct_exact_int(self):
        """Float that is exactly integer should work."""
        result = discrim(80.0, 100, method="triangle")
        assert np.isfinite(result.d_prime)

    def test_float_correct_non_int_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            discrim(80.5, 100, method="triangle")

    def test_float_total_exact_int(self):
        result = discrim(80, 100.0, method="triangle")
        assert np.isfinite(result.d_prime)

    def test_float_total_non_int_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            discrim(80, 100.5, method="triangle")

    def test_negative_correct_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            discrim(-1, 100, method="triangle")

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            discrim(0, 0, method="triangle")

    def test_correct_greater_than_total_raises(self):
        with pytest.raises(ValueError, match="cannot be larger"):
            discrim(101, 100, method="triangle")

    def test_invalid_conf_level_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            discrim(80, 100, conf_level=1.0)

    def test_invalid_test_type_raises(self):
        with pytest.raises(ValueError, match="Unknown test"):
            discrim(80, 100, test="bogus")


class TestDiscrimProtocols:
    """Test across all protocols."""

    @pytest.mark.parametrize("method", [
        "triangle", "twoafc", "duotrio", "threeafc", "tetrad", "hexad", "twofive",
    ])
    def test_all_protocols(self, method):
        result = discrim(70, 100, method=method)
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)
        assert 0 <= result.pc <= 1
        assert 0 <= result.pd <= 1
