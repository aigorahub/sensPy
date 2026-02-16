"""Additional tests for senspy.links module to improve coverage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from senspy.links.psychometric import (
    get_link,
    psy_fun,
    psy_inv,
    psy_deriv,
    twoafc_link,
    duotrio_link,
    triangle_link,
    threeafc_link,
    tetrad_link,
    hexad_link,
    twofive_link,
    twofivef_link,
    _duotrio_linkfun,
    _triangle_linkfun,
    _threeafc_linkfun,
    _tetrad_linkfun,
    _hexad_linkfun,
    _twofive_linkfun,
    _twofivef_linkfun,
)


class TestLinkfunBoundaryConditions:
    """Test boundary conditions in linkfun (inverse) functions."""

    def test_duotrio_linkfun_at_chance(self):
        """pc <= p_guess should return d'=0."""
        result = _duotrio_linkfun(np.array([0.5]))
        assert result[0] == 0.0

    def test_duotrio_linkfun_below_chance(self):
        result = _duotrio_linkfun(np.array([0.3]))
        assert result[0] == 0.0

    def test_duotrio_linkfun_near_one(self):
        """pc near 1 should return inf."""
        result = _duotrio_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_triangle_linkfun_at_chance(self):
        result = _triangle_linkfun(np.array([1 / 3]))
        assert result[0] == 0.0

    def test_triangle_linkfun_below_chance(self):
        result = _triangle_linkfun(np.array([0.2]))
        assert result[0] == 0.0

    def test_triangle_linkfun_near_one(self):
        result = _triangle_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_threeafc_linkfun_at_chance(self):
        result = _threeafc_linkfun(np.array([1 / 3]))
        assert result[0] == 0.0

    def test_threeafc_linkfun_near_one(self):
        result = _threeafc_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_tetrad_linkfun_at_chance(self):
        result = _tetrad_linkfun(np.array([1 / 3]))
        assert result[0] == 0.0

    def test_tetrad_linkfun_near_one(self):
        result = _tetrad_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_hexad_linkfun_at_chance(self):
        result = _hexad_linkfun(np.array([0.1]))
        assert result[0] == 0.0

    def test_hexad_linkfun_below_chance(self):
        result = _hexad_linkfun(np.array([0.05]))
        assert result[0] == 0.0

    def test_hexad_linkfun_near_one(self):
        result = _hexad_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_twofive_linkfun_at_chance(self):
        result = _twofive_linkfun(np.array([0.1]))
        assert result[0] == 0.0

    def test_twofive_linkfun_near_one(self):
        result = _twofive_linkfun(np.array([1.0]))
        assert result[0] == np.inf

    def test_twofivef_linkfun_at_chance(self):
        result = _twofivef_linkfun(np.array([0.4]))
        assert result[0] == 0.0

    def test_twofivef_linkfun_near_one(self):
        result = _twofivef_linkfun(np.array([1.0]))
        assert result[0] == np.inf


class TestLinkinvBoundaryConditions:
    """Test boundary conditions in linkinv functions."""

    def test_hexad_linkinv_at_zero(self):
        result = hexad_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 0.1)

    def test_hexad_linkinv_very_large(self):
        """Very large d' should give ~1."""
        result = hexad_link.linkinv(np.array([15.0]))
        assert_allclose(result[0], 1.0)

    def test_twofive_linkinv_at_zero(self):
        result = twofive_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 0.1)

    def test_twofive_linkinv_very_large(self):
        result = twofive_link.linkinv(np.array([15.0]))
        assert_allclose(result[0], 1.0)

    def test_twofivef_linkinv_at_zero(self):
        result = twofivef_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 0.4)


class TestMuEtaBoundary:
    """Test mu_eta (derivative) at boundaries."""

    def test_twoafc_mu_eta_at_zero(self):
        """Derivative at d'=0 should be positive for twoafc."""
        result = twoafc_link.mu_eta(np.array([0.0]))
        assert result[0] > 0

    def test_duotrio_mu_eta_at_zero(self):
        result = duotrio_link.mu_eta(np.array([0.0]))
        assert result[0] >= 0

    def test_triangle_mu_eta_at_zero(self):
        result = triangle_link.mu_eta(np.array([0.0]))
        assert result[0] >= 0

    def test_tetrad_mu_eta_at_zero(self):
        """d' < h returns 0."""
        result = tetrad_link.mu_eta(np.array([0.0]))
        assert result[0] == 0.0

    def test_tetrad_mu_eta_positive(self):
        result = tetrad_link.mu_eta(np.array([1.0]))
        assert result[0] > 0

    def test_hexad_mu_eta_at_zero(self):
        result = hexad_link.mu_eta(np.array([0.0]))
        assert result[0] == 0.0

    def test_hexad_mu_eta_positive(self):
        result = hexad_link.mu_eta(np.array([3.0]))
        assert result[0] >= 0

    def test_twofive_mu_eta_at_zero(self):
        result = twofive_link.mu_eta(np.array([0.0]))
        assert result[0] == 0.0

    def test_twofive_mu_eta_positive(self):
        result = twofive_link.mu_eta(np.array([1.0]))
        assert result[0] > 0

    def test_twofivef_mu_eta_at_zero(self):
        result = twofivef_link.mu_eta(np.array([0.0]))
        assert result[0] == 0.0

    def test_twofivef_mu_eta_positive(self):
        result = twofivef_link.mu_eta(np.array([1.0]))
        assert result[0] > 0

    def test_threeafc_mu_eta_at_zero(self):
        """Derivative at d'=0 should be 0 (defined explicitly)."""
        result = threeafc_link.mu_eta(np.array([0.0]))
        assert result[0] == 0.0

    def test_threeafc_mu_eta_positive(self):
        result = threeafc_link.mu_eta(np.array([1.5]))
        assert result[0] > 0


class TestLinkinvConsistency:
    """Test linkinv and linkfun are inverses."""

    @pytest.mark.parametrize("link,d_vals", [
        (duotrio_link, [0.5, 1.0, 2.0]),
        (triangle_link, [0.5, 1.0, 2.0]),
        (twoafc_link, [0.5, 1.0, 2.0]),
        (threeafc_link, [0.5, 1.0, 2.0]),
        (tetrad_link, [0.5, 1.0, 2.0]),
    ])
    def test_roundtrip(self, link, d_vals):
        """linkfun(linkinv(d)) should recover d."""
        for d in d_vals:
            pc = link.linkinv(np.array([d]))
            d_recovered = link.linkfun(pc)
            assert_allclose(d_recovered[0], d, rtol=1e-3)

    def test_hexad_roundtrip(self):
        """Hexad roundtrip at moderate d'."""
        for d in [2.0, 3.0, 4.0]:
            pc = hexad_link.linkinv(np.array([d]))
            if pc[0] > 0.1 + 1e-6 and pc[0] < 1.0 - 1e-6:
                d_recovered = hexad_link.linkfun(pc)
                assert_allclose(d_recovered[0], d, rtol=0.1)

    def test_twofive_roundtrip(self):
        for d in [0.5, 1.0]:
            pc = twofive_link.linkinv(np.array([d]))
            d_recovered = twofive_link.linkfun(pc)
            assert_allclose(d_recovered[0], d, rtol=5e-2)

    def test_twofivef_roundtrip(self):
        for d in [0.5, 1.0, 2.0]:
            pc = twofivef_link.linkinv(np.array([d]))
            d_recovered = twofivef_link.linkfun(pc)
            assert_allclose(d_recovered[0], d, rtol=1e-2)


class TestPsyFunInvDeriv:
    """Tests for psy_fun, psy_inv, psy_deriv top-level functions."""

    def test_psy_fun_negative_d_prime_raises(self):
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            psy_fun(-1.0, method="triangle")

    def test_psy_inv_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pc must be in"):
            psy_inv(-0.1, method="triangle")

    def test_psy_deriv_negative_raises(self):
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            psy_deriv(-1.0, method="triangle")

    def test_psy_fun_array(self):
        result = psy_fun([0, 1, 2], method="twoafc")
        assert len(result) == 3
        assert result[0] == pytest.approx(0.5)

    def test_psy_inv_array(self):
        result = psy_inv([0.5, 0.7, 0.9], method="twoafc")
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_psy_deriv_array(self):
        result = psy_deriv([0, 1, 2], method="twoafc")
        assert len(result) == 3

    @pytest.mark.parametrize("method", [
        "triangle", "twoafc", "duotrio", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef",
    ])
    def test_psy_fun_monotonic(self, method):
        """psy_fun should be monotonically increasing."""
        d_vals = np.array([0, 0.5, 1, 1.5, 2, 3])
        pc = psy_fun(d_vals, method=method)
        assert np.all(np.diff(pc) >= -1e-10)

    @pytest.mark.parametrize("method", [
        "triangle", "twoafc", "duotrio", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef",
    ])
    def test_psy_fun_at_zero_equals_p_guess(self, method):
        """psy_fun(0) should equal p_guess."""
        link = get_link(method)
        pc = psy_fun(0, method=method)
        assert_allclose(pc[0], link.p_guess, rtol=1e-2)

    @pytest.mark.parametrize("method", [
        "triangle", "twoafc", "duotrio", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef",
    ])
    def test_psy_deriv_nonnegative(self, method):
        """Derivative should always be non-negative."""
        d_vals = np.array([0, 0.5, 1.0, 2.0, 3.0])
        deriv = psy_deriv(d_vals, method=method)
        assert np.all(deriv >= 0)


class TestGetLink:
    """Test get_link registry."""

    @pytest.mark.parametrize("method", [
        "triangle", "twoafc", "duotrio", "threeafc",
        "tetrad", "hexad", "twofive", "twofivef",
    ])
    def test_all_methods(self, method):
        link = get_link(method)
        assert link.name == method
        assert hasattr(link, "linkinv")
        assert hasattr(link, "linkfun")
        assert hasattr(link, "mu_eta")
        assert 0 < link.p_guess < 1


class TestLinkVectorized:
    """Test that links work with vector inputs."""

    @pytest.mark.parametrize("link", [
        twoafc_link, duotrio_link, triangle_link, threeafc_link,
        tetrad_link, hexad_link, twofive_link, twofivef_link,
    ])
    def test_linkinv_vectorized(self, link):
        d_vals = np.array([0.0, 0.5, 1.0, 2.0])
        result = link.linkinv(d_vals)
        assert result.shape == d_vals.shape
        assert np.all(result >= link.p_guess - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    @pytest.mark.parametrize("link", [
        twoafc_link, duotrio_link, triangle_link, threeafc_link,
        tetrad_link,
    ])
    def test_mu_eta_vectorized(self, link):
        d_vals = np.array([0.5, 1.0, 2.0])
        result = link.mu_eta(d_vals)
        assert result.shape == d_vals.shape
        assert np.all(result >= 0)


class TestLinkSpecificValues:
    """Test specific known values for protocols."""

    def test_twoafc_at_sqrt2(self):
        """At d'=sqrt(2), Phi(1)=0.8413."""
        from scipy import stats
        d = np.sqrt(2)
        expected = stats.norm.cdf(1.0)
        result = twoafc_link.linkinv(np.array([d]))
        assert_allclose(result[0], expected, rtol=1e-6)

    def test_triangle_at_zero(self):
        result = triangle_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 1/3)

    def test_duotrio_at_zero(self):
        result = duotrio_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 0.5)

    def test_threeafc_at_zero(self):
        result = threeafc_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 1/3)

    def test_tetrad_at_zero(self):
        result = tetrad_link.linkinv(np.array([0.0]))
        assert_allclose(result[0], 1/3)
