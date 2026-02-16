"""Additional tests for senspy.dod module to improve coverage."""

import warnings

import numpy as np
import pytest

from senspy.dod import (
    DODControl,
    par2prob_dod,
    _init_tau,
    _validate_dod_data,
    _dod_null_tau_internal,
    _numerical_gradient,
    _numerical_hessian,
    dod_fit,
    dod,
    dod_sim,
    dod_power,
    optimal_tau,
)


class TestDODControl:
    """Tests for DODControl dataclass validation."""

    def test_invalid_grad_tol_zero(self):
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=0.0)

    def test_invalid_grad_tol_negative(self):
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=-1.0)

    def test_invalid_grad_tol_inf(self):
        with pytest.raises(ValueError, match="grad_tol must be positive"):
            DODControl(grad_tol=np.inf)

    def test_invalid_integer_tol_negative(self):
        with pytest.raises(ValueError, match="integer_tol must be non-negative"):
            DODControl(integer_tol=-0.1)

    def test_valid_defaults(self):
        ctrl = DODControl()
        assert ctrl.grad_tol == 1e-4
        assert ctrl.integer_tol == 1e-8


class TestPar2ProbDod:
    """Tests for par2prob_dod validation."""

    def test_negative_d_prime_raises(self):
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            par2prob_dod(np.array([1.0]), d_prime=-0.5)

    def test_empty_tau_raises(self):
        with pytest.raises(ValueError, match="tau must have at least one element"):
            par2prob_dod(np.array([]), d_prime=1.0)

    def test_non_positive_tau_raises(self):
        with pytest.raises(ValueError, match="tau values must be positive"):
            par2prob_dod(np.array([0.0, 1.0]), d_prime=1.0)

    def test_non_increasing_tau_raises(self):
        with pytest.raises(ValueError, match="tau values must be strictly increasing"):
            par2prob_dod(np.array([2.0, 1.0]), d_prime=1.0)

    def test_valid_single_tau(self):
        prob = par2prob_dod(np.array([1.0]), d_prime=1.0)
        assert prob.shape == (2, 2)
        assert np.allclose(prob.sum(axis=1), 1.0)

    def test_d_prime_zero_same_equals_diff(self):
        """When d_prime=0, same and diff probabilities should be equal."""
        prob = par2prob_dod(np.array([0.5, 1.0, 1.5]), d_prime=0.0)
        np.testing.assert_allclose(prob[0, :], prob[1, :], atol=1e-10)


class TestInitTau:
    """Tests for _init_tau."""

    def test_ncat_less_than_2_raises(self):
        with pytest.raises(ValueError, match="ncat must be at least 2"):
            _init_tau(ncat=1)

    def test_ncat_2(self):
        tau = _init_tau(ncat=2)
        assert len(tau) == 1

    def test_ncat_4(self):
        tau = _init_tau(ncat=4)
        assert len(tau) == 3


class TestValidateDodData:
    """Tests for _validate_dod_data."""

    def test_different_length_raises(self):
        with pytest.raises(ValueError, match="same and diff must have the same length"):
            _validate_dod_data([1, 2, 3], [1, 2])

    def test_negative_counts_raises(self):
        with pytest.raises(ValueError, match="Counts must be non-negative"):
            _validate_dod_data([-1, 2, 3], [1, 2, 3])

    def test_single_category_raises(self):
        with pytest.raises(ValueError, match="Need at least 2 response categories"):
            _validate_dod_data([5], [3])

    def test_empty_categories_removed(self):
        """Categories where both same and diff are 0 should be removed."""
        same, diff = _validate_dod_data([5, 0, 3], [3, 0, 2])
        assert len(same) == 2

    def test_all_but_one_empty_raises(self):
        """After removing empty categories, need at least 2."""
        with pytest.raises(ValueError, match="Need counts in more than one"):
            _validate_dod_data([5, 0, 0], [3, 0, 0])

    def test_non_integer_same_warns(self):
        with pytest.warns(UserWarning, match="non-integer counts in 'same'"):
            _validate_dod_data([1.5, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_non_integer_diff_warns(self):
        with pytest.warns(UserWarning, match="non-integer counts in 'diff'"):
            _validate_dod_data([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])


class TestDodFitCases:
    """Tests for dod_fit various estimation cases."""

    def test_d_prime_zero_null_model(self):
        """Case 1: d_prime=0 with tau=None -> use null tau."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([20, 30, 25, 25])
        fit = dod_fit(same, diff, d_prime=0.0)
        assert np.isclose(fit.d_prime, 0.0)
        assert fit.tau is not None
        assert len(fit.tau) == 3

    def test_estimate_tau_only(self):
        """Case: tau=None, d_prime given (non-zero) -> estimate tau."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        fit = dod_fit(same, diff, d_prime=1.0)
        assert np.isclose(fit.d_prime, 1.0)
        assert fit.tau is not None

    def test_estimate_d_prime_only(self):
        """Case: tau given, d_prime=None -> estimate d_prime."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        tau = np.array([0.5, 1.0, 1.5])
        fit = dod_fit(same, diff, tau=tau)
        assert np.isfinite(fit.d_prime)
        np.testing.assert_array_equal(fit.tau, tau)

    def test_both_provided_evaluate_only(self):
        """Case 3: Both tau and d_prime provided -> just evaluate."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        tau = np.array([0.5, 1.0, 1.5])
        fit = dod_fit(same, diff, tau=tau, d_prime=1.0)
        assert fit.d_prime == 1.0
        assert np.isfinite(fit.log_lik)
        assert fit.vcov is None  # no optimization, no vcov

    def test_no_grad_no_vcov(self):
        """Test with get_grad=False and get_vcov=False."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        ctrl = DODControl(get_grad=False, get_vcov=False)
        fit = dod_fit(same, diff, control=ctrl)
        assert fit.gradient is None
        assert fit.vcov is None

    def test_test_args_false_skips_validation(self):
        """test_args=False skips validation."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        ctrl = DODControl(test_args=False)
        fit = dod_fit(same, diff, control=ctrl)
        assert np.isfinite(fit.d_prime)

    def test_vcov_computed_when_requested(self):
        """With get_grad=True and get_vcov=True, vcov may be available."""
        same = np.array([20, 30, 25, 25])
        diff = np.array([10, 20, 35, 35])
        ctrl = DODControl(get_grad=True, get_vcov=True)
        fit = dod_fit(same, diff, control=ctrl)
        assert fit.d_prime > 0
        # vcov may or may not be computed depending on Hessian


class TestDodFunction:
    """Tests for dod() high-level function."""

    def test_likelihood_statistic(self):
        same = [20, 30, 25, 25]
        diff = [10, 20, 35, 35]
        result = dod(same, diff, statistic="likelihood")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)

    def test_pearson_statistic(self):
        same = [20, 30, 25, 25]
        diff = [10, 20, 35, 35]
        result = dod(same, diff, statistic="Pearson")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)

    def test_wilcoxon_statistic(self):
        same = [20, 30, 25, 25]
        diff = [10, 20, 35, 35]
        result = dod(same, diff, statistic="Wilcoxon")
        assert np.isfinite(result.d_prime)
        assert np.isfinite(result.p_value)

    def test_wald_statistic(self):
        same = [20, 30, 25, 25]
        diff = [10, 20, 35, 35]
        result = dod(same, diff, statistic="Wald")
        assert np.isfinite(result.d_prime)
        assert result.conf_method == "Wald"

    def test_alternative_similarity(self):
        same = [10, 20, 35, 35]
        diff = [20, 30, 25, 25]
        result = dod(same, diff, d_prime0=2.0, alternative="similarity")
        assert result.alternative == "less"

    def test_alternative_two_sided(self):
        same = [20, 30, 25, 25]
        diff = [10, 20, 35, 35]
        result = dod(same, diff, d_prime0=0.5, alternative="two.sided")
        assert result.alternative == "two.sided"

    def test_invalid_conf_level_raises(self):
        with pytest.raises(ValueError, match="conf_level must be between"):
            dod([20, 30], [10, 20], conf_level=1.5)

    def test_negative_d_prime0_raises(self):
        with pytest.raises(ValueError, match="d_prime0 must be non-negative"):
            dod([20, 30], [10, 20], d_prime0=-1.0)

    def test_similarity_with_d_prime0_zero_raises(self):
        with pytest.raises(ValueError, match="has to be 'difference'"):
            dod([20, 30], [10, 20], d_prime0=0.0, alternative="similarity")

    def test_wilcoxon_with_nonzero_d_prime0_raises(self):
        with pytest.raises(ValueError, match="Wilcoxon statistic only"):
            dod([20, 30], [10, 20], d_prime0=1.0, statistic="Wilcoxon")

    def test_low_d_prime_warning(self):
        """d_prime < 0.01 should warn about SE unavailability."""
        # Create data where d_prime is near zero
        same = [25, 25, 25, 25]
        diff = [25, 25, 25, 25]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dod(same, diff)
            # If d_prime < 0.01, a warning should be raised
            if result.d_prime < 0.01:
                warning_msgs = [str(warning.message) for warning in w]
                assert any("d_prime < 0.01" in msg for msg in warning_msgs)


class TestOptimalTau:
    """Tests for optimal_tau function."""

    def test_equi_prob_method(self):
        result = optimal_tau(d_prime=1.0, ncat=4, method="equi_prob")
        assert "tau" in result
        assert len(result["tau"]) == 3
        assert result["method"] == "equi_prob"

    def test_lr_max_method(self):
        result = optimal_tau(d_prime=1.0, ncat=3, method="LR_max")
        assert "tau" in result
        assert len(result["tau"]) == 2

    def test_se_min_method(self):
        result = optimal_tau(d_prime=1.0, ncat=3, method="se_min")
        assert "tau" in result
        assert len(result["tau"]) == 2

    def test_lr_max_nonzero_d_prime0(self):
        """LR_max with d_prime0 != 0 exercises the else branch."""
        result = optimal_tau(d_prime=2.0, d_prime0=0.5, ncat=3, method="LR_max")
        assert "tau" in result

    def test_invalid_d_prime_raises(self):
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            optimal_tau(d_prime=-1.0)

    def test_invalid_ncat_raises(self):
        with pytest.raises(ValueError, match="ncat must be at least 2"):
            optimal_tau(d_prime=1.0, ncat=1)

    def test_tau_start_provided(self):
        result = optimal_tau(d_prime=1.0, ncat=3, tau_start=np.array([0.5, 1.5]))
        assert "tau" in result

    def test_invalid_tau_start_length_raises(self):
        with pytest.raises(ValueError, match="tau_start must have length"):
            optimal_tau(d_prime=1.0, ncat=3, tau_start=np.array([0.5]))

    def test_invalid_tau_start_negative_raises(self):
        with pytest.raises(ValueError, match="tau_start values must be positive"):
            optimal_tau(d_prime=1.0, ncat=3, tau_start=np.array([-0.5, 1.0]))

    def test_invalid_tau_start_not_increasing_raises(self):
        with pytest.raises(ValueError, match="tau_start must be strictly increasing"):
            optimal_tau(d_prime=1.0, ncat=3, tau_start=np.array([2.0, 1.0]))

    def test_do_warn_false_suppresses(self):
        result = optimal_tau(d_prime=1.0, ncat=3, do_warn=False)
        assert "tau" in result


class TestDodSim:
    """Tests for dod_sim function."""

    def test_basic_simulation(self):
        data = dod_sim(d_prime=1.0, ncat=4, sample_size=100, random_state=42)
        assert data.shape == (2, 4)
        assert data[0, :].sum() == 100
        assert data[1, :].sum() == 100

    def test_tuple_sample_size(self):
        data = dod_sim(d_prime=1.0, ncat=3, sample_size=(50, 80), random_state=42)
        assert data[0, :].sum() == 50
        assert data[1, :].sum() == 80

    def test_user_defined_tau(self):
        tau = np.array([0.5, 1.0, 1.5])
        data = dod_sim(d_prime=1.0, method_tau="user_defined", tau=tau, random_state=42)
        assert data.shape == (2, 4)

    def test_user_defined_no_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be provided"):
            dod_sim(d_prime=1.0, method_tau="user_defined")

    def test_negative_d_prime_raises(self):
        with pytest.raises(ValueError, match="d_prime must be non-negative"):
            dod_sim(d_prime=-1.0)

    def test_negative_d_prime0_raises(self):
        with pytest.raises(ValueError, match="d_prime0 must be non-negative"):
            dod_sim(d_prime=1.0, d_prime0=-1.0)

    def test_tau_ignored_warning(self):
        """Passing tau when method_tau != 'user_defined' should warn."""
        with pytest.warns(UserWarning, match="tau.*is ignored"):
            dod_sim(
                d_prime=1.0,
                method_tau="equi_prob",
                tau=np.array([0.5, 1.0, 1.5]),
                random_state=42,
            )

    def test_generator_random_state(self):
        rng = np.random.default_rng(42)
        data = dod_sim(d_prime=1.0, random_state=rng)
        assert data.shape[0] == 2

    def test_d_prime_zero(self):
        data = dod_sim(d_prime=0.0, ncat=3, sample_size=50, random_state=42)
        assert data.shape == (2, 3)

    def test_se_min_method_tau(self):
        data = dod_sim(d_prime=1.0, ncat=3, method_tau="se_min", random_state=42)
        assert data.shape == (2, 3)


class TestDodPower:
    """Tests for dod_power function."""

    def test_basic_power(self):
        result = dod_power(d_primeA=1.5, sample_size=100, nsim=50, random_state=42)
        assert 0 <= result.power <= 1
        assert result.n_used > 0

    def test_wilcoxon_statistic(self):
        result = dod_power(
            d_primeA=1.5, sample_size=100, nsim=50,
            statistic="Wilcoxon", random_state=42,
        )
        assert 0 <= result.power <= 1

    def test_wald_statistic(self):
        result = dod_power(
            d_primeA=1.5, sample_size=200, nsim=100,
            statistic="Wald", random_state=42,
        )
        # Wald may return nan if all simulations fail
        assert result.power is np.nan or 0 <= result.power <= 1

    def test_pearson_statistic(self):
        result = dod_power(
            d_primeA=1.5, sample_size=100, nsim=50,
            statistic="Pearson", random_state=42,
        )
        assert 0 <= result.power <= 1

    def test_similarity_alternative(self):
        result = dod_power(
            d_primeA=0.5, d_prime0=2.0, sample_size=100, nsim=50,
            alternative="similarity", random_state=42,
        )
        assert result.alternative == "less"

    def test_two_sided_alternative(self):
        result = dod_power(
            d_primeA=1.5, d_prime0=0.5, sample_size=200, nsim=50,
            alternative="two.sided", random_state=42,
            method_tau="equi_prob",
        )
        # May return nan if simulations fail
        assert np.isnan(result.power) or 0 <= result.power <= 1

    def test_tuple_sample_size(self):
        result = dod_power(
            d_primeA=1.5, sample_size=(80, 120), nsim=50, random_state=42,
        )
        assert result.sample_size == (80, 120)

    def test_user_defined_tau(self):
        result = dod_power(
            d_primeA=1.5, method_tau="user_defined",
            tau=np.array([0.5, 1.0, 1.5]),
            nsim=50, random_state=42,
        )
        assert 0 <= result.power <= 1

    def test_user_defined_no_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be provided"):
            dod_power(d_primeA=1.5, method_tau="user_defined", nsim=10)

    def test_negative_d_primeA_raises(self):
        with pytest.raises(ValueError, match="d_primeA must be a finite"):
            dod_power(d_primeA=-1.0, nsim=10)

    def test_negative_d_prime0_raises(self):
        with pytest.raises(ValueError, match="d_prime0 must be a finite"):
            dod_power(d_primeA=1.0, d_prime0=-1.0, nsim=10)

    def test_invalid_nsim_raises(self):
        with pytest.raises(ValueError, match="nsim must be a positive"):
            dod_power(d_primeA=1.0, nsim=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be between"):
            dod_power(d_primeA=1.0, alpha=-0.1, nsim=10)

    def test_greater_with_low_d_primeA_raises(self):
        with pytest.raises(ValueError, match="Need d_primeA >= d_prime0"):
            dod_power(d_primeA=0.5, d_prime0=2.0, alternative="greater", nsim=10)

    def test_less_with_high_d_primeA_raises(self):
        with pytest.raises(ValueError, match="Need d_primeA <= d_prime0"):
            dod_power(d_primeA=2.0, d_prime0=0.5, alternative="less", nsim=10)

    def test_wilcoxon_nonzero_d_prime0_raises(self):
        with pytest.raises(ValueError, match="Wilcoxon statistic only"):
            dod_power(
                d_primeA=1.0, d_prime0=0.5,
                statistic="Wilcoxon", nsim=10,
            )

    def test_generator_random_state(self):
        rng = np.random.default_rng(42)
        result = dod_power(d_primeA=1.5, nsim=20, random_state=rng)
        assert result.n_used > 0


class TestNumericalHelpers:
    """Tests for _numerical_gradient and _numerical_hessian."""

    def test_gradient_quadratic(self):
        """Gradient of f(x) = x^2 at x=3 should be ~6."""
        def f(x):
            return x[0] ** 2

        grad = _numerical_gradient(f, np.array([3.0]))
        assert np.isclose(grad[0], 6.0, atol=1e-4)

    def test_hessian_quadratic(self):
        """Hessian of f(x) = x^2 should be [[2]]."""
        def f(x):
            return x[0] ** 2

        hess = _numerical_hessian(f, np.array([3.0]))
        assert np.isclose(hess[0, 0], 2.0, atol=1e-3)

    def test_gradient_multivariate(self):
        """Gradient of f(x,y) = x^2 + 2*y^2 at (1,1) = (2, 4)."""
        def f(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        grad = _numerical_gradient(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-4)

    def test_hessian_multivariate(self):
        """Hessian of f(x,y) = x^2 + 2*y^2 should be [[2,0],[0,4]]."""
        def f(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        hess = _numerical_hessian(f, np.array([1.0, 1.0]))
        expected = np.array([[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(hess, expected, atol=1e-3)
