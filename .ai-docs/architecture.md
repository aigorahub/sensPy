# aigorahub/sensPy - AI Documentation

*Generated: 2026-02-16*

---

## Executive Summary

SensPy is a specialized Python 3.10+ library designed for sensory discrimination analysis, serving as a high-performance port of the R sensR package. It implements Thurstonian models to estimate sensitivity indices like d-prime and probabilities of discrimination across various protocols including Triangle, Duo-Trio, and 2-AFC. The architecture prioritizes numerical parity with R, utilizing SciPy for optimization and Numba for performance-critical calculations. An AI agent should prioritize understanding the strict validation requirements established in the testing suite. The project uses modern tooling like Poetry and uv for dependency management, and GitHub Actions for AI-augmented code reviews. Success in this codebase depends on adhering to the established functional patterns and ensuring all statistical outputs align with the 'golden data' fixtures derived from the original R implementation.

## Introduction

This documentation provides a comprehensive guide to the SensPy codebase, covering core API implementation, statistical modules, and the validation framework. It is intended to help developers and AI agents navigate the library's multi-layered approach to sensory science, from raw data processing to complex power simulations. Key concepts include Thurstonian modeling, which provides the psychological framework for discrimination tasks, and the distinction between analytical and empirical power analysis. Users should leverage the cross-module relationships to understand how core functions are exposed and validated effectively across the senspy and tests modules.

## Table of Contents

1. [Project Setup and Root Files](#project-setup-and-root-files)
2. [SensPy Core API Implementation](#senspy-core-api-implementation)
3. [Development Utilities and Metadata](#development-utilities-and-metadata)
4. [Specific Statistical Tests Modules](#specific-statistical-tests-modules)
5. [Visualization and Data Utility](#visualization-and-data-utility)
6. [Testing and Validation Suite](#testing-and-validation-suite)
7. [Simulation and Power Analysis](#simulation-and-power-analysis)

---

# Project Setup and Root Files

## Overview
The `sensPy` project is a Python port of the R package `sensR`, focused on Thurstonian models for sensory discrimination analysis. The setup provides a modern Python 3.10+ environment using Poetry for dependency management and GitHub Actions for a comprehensive CI/CD pipeline. The root configuration ensures numerical parity with R, high performance via Numba, and automated quality assurance through linting and AI-assisted reviews.

## Key Components

### Project Configuration
- **pyproject.toml**: The source of truth for the project. It defines the build system (Poetry), core dependencies (NumPy, SciPy, Pandas, Plotly, Numba), and configuration for development tools like `ruff`, `black`, `mypy`, and `pytest`.
- **README.md**: Acts as the primary documentation entry point, detailing supported protocols (Triangle, Duo-Trio, 2-AFC, etc.), statistical models (`discrim`, `betabin`, `samediff`), and installation procedures.
- **CONTRIBUTING.md**: Establishes the development workflow, requiring PEP 8 compliance, NumPy-style docstrings, and mandatory numerical validation against R outputs.

### Continuous Integration (.github/workflows)
- **Automated Testing**: `pytest.yml` handles the execution of over 500 tests to ensure library stability.
- **AI Code Review**: Unique workflows like `auto-comment-gemini-review.yml` and `claude-code-review.yml` automate feedback by triggering AI reviews on pull requests using GitHub API calls.
- **Documentation & Publishing**: `docs.yml` and `publish.yml` automate MkDocs deployment and package distribution.

## Data Flow
Project configuration flows from `pyproject.toml` into the local development environment via Poetry. When code is pushed, GitHub Actions orchestrates a multi-stage flow:
1. **Validation**: Linting (`ruff`), formatting (`black`), and type checking (`mypy`).
2. **Testing**: Execution of unit and parity tests via `pytest`.
3. **Review**: Automated AI comments are generated for open PRs to assist maintainers.
4. **Deployment**: Successful merges to main trigger documentation updates and package builds.

## Dependencies
- **Numerical/Statistical**: `numpy` (>=1.23), `scipy` (>=1.9), `pandas` (>=1.5), and `numba` (>=0.56) for performance-critical Thurstonian calculations.
- **Visualization**: `plotly` (>=5.15) is the default for interactive plots, while `matplotlib` (>=3.6) is an optional extra (`static-plots`).
- **Interoperability**: `rpy2` is an optional group dependency used specifically for validating parity against the original R implementation.

## Patterns and Conventions
- **Numerical Parity**: A strict requirement where all outputs must match `sensR` results. This is reflected in the test suite markers (e.g., `@pytest.mark.rpy2`).
- **Modern Tooling**: Use of `ruff` for fast linting and `mypy` for static analysis ensures a high-quality codebase.
- **Porting Roadmap**: The project uses specialized documentation like `PORTING_PLAN.md` and `GAP_ANALYSIS.md` (referenced in CONTRIBUTING.md) to track progress against the R original.

## Gotchas
- **Numba Requirement**: The project relies on `numba`, which requires a compatible LLVM version on the host system.
- **R Dependency for Tests**: While `senspy` is a Python library, contributors running the full test suite may need R and `sensR` installed to pass `rpy2` validation tests.
- **Optional Plotting**: Standard `pip install senspy` does not include `matplotlib`. Users requiring static plots must install the `static-plots` extra.

> **Notes:**
> - The project utilizes Numba for high-performance numerical execution of Thurstonian models.
> - CI/CD includes advanced automation that triggers AI-based reviews via GitHub Actions and the Gemini API.
> - Project structure enforces numerical parity with the R 'sensR' package as a primary correctness metric.
> - Dependency management is strictly handled through Poetry, separating core, dev, and optional rpy2 groups.
> - Manual validation against R is required for new features to maintain the port's integrity.
> - Plotting functionality is split between Plotly (core) and Matplotlib (optional), which may lead to ImportErrors if extras are not installed.
> - The 'auto-comment-gemini-review' workflow requires a 'GEMINI_REVIEW_TOKEN' secret to function in forks or local environments.

---

# SensPy Core API Implementation

## Overview
This section documents the primary entry points and statistical engines of the `senspy` package. It implements Thurstonian models for sensory discrimination, providing tools to estimate sensitivity ($d'$), probability of discrimination ($P_d$), and probability of correct response ($P_c$). The implementation follows a porting strategy from the R `sensR` package, leveraging `scipy` for optimization and statistical distributions to handle sensory protocols ranging from simple Triangle tests to complex Same-Different and 2-AC models.

## Key Components

### Primary Analysis Functions
- **`senspy.discrim()`**: The main entry point for standard forced-choice protocols (Triangle, Duo-Trio, n-AFC). It supports four test statistics: `exact` (Binomial), `likelihood` (Signed Likelihood Root), `wald`, and `score` (Wilson).
- **`senspy.betabin.betabin()`**: Implements beta-binomial models to account for overdispersion in replicated discrimination data. It supports both standard and chance-corrected models where $\mu$ represents $P_d$.
- **`senspy.anota.anota()`**: A specialized implementation of the A-Not-A protocol using probit regression ($z(H) - z(F)$) and Fisher's Exact Test for significance.

### Hypothesis Testing
- **`senspy.dprime_tests.dprime_test()`**: Tests if a common $d'$ across multiple groups equals a specific null value ($d'_0$).
- **`senspy.dprime_tests.dprime_compare()`**: Performs an 'any-difference' test (Chi-square) across multiple groups to see if they share a common $d'$.
- **`senspy.dprime_tests.posthoc()`**: Conducts pairwise comparisons between groups with p-value adjustments (`holm`, `bonferroni`) and compact letter displays (CLD).

### Power and Simulation
- **`senspy.power`**: Contains analytical power and sample size calculations (`discrim_power`, `dprime_sample_size`) using normal approximations or exact binomial search.
- **`senspy.protocol_power`**: Handles complex protocols where analytical solutions are difficult. `samediff_power` uses Monte Carlo simulation, while `twoac_power` uses exact enumeration of outcomes up to $N=5000$.
- **`senspy.simulation`**: Provides `discrim_sim` (with optional `sd_indiv` for overdispersion) and `samediff_sim` for generating synthetic sensory data.

## Data Flow
1. **Input**: Functions typically accept raw counts (`correct`, `total`) or NumPy arrays for multi-group data, along with a `Protocol` identifier (e.g., 'triangle').
2. **Parameter Mapping**: For simple protocols, `senspy.links` maps $d'$ to $P_c$ and vice versa. For complex models (DOD, 2-AC), internal NLL (Negative Log-Likelihood) functions are defined.
3. **Optimization**: Most models use `scipy.optimize.minimize` or `minimize_scalar`. Boundary constraints are strictly enforced ($d' \geq 0$, $\mu \in [0, 1]$).
4. **Standard Errors**: Computed via the Delta Method (for $d'$) or the inverse Hessian (for beta-binomial models).
5. **Results**: Data is returned in specialized dataclasses (e.g., `DiscrimResult`, `BetaBinomialResult`) that provide formatting methods and derived properties like confidence intervals.

## Patterns Used
- **Result Dataclasses**: All analysis functions return immutable-style dataclasses containing estimates, standard errors, and metadata about the optimizer's convergence.
- **Link Function Abstraction**: Protocol-specific logic is encapsulated in link functions (`psy_fun`, `psy_inv`, `psy_deriv`), allowing `discrim()` to remain protocol-agnostic.
- **Standardized Naming**: Follows `sensR` conventions (e.g., `pd` for probability of discrimination) while adopting Pythonic snake_case.
- **Signed Likelihood Root**: Frequently used ($z = \text{sign}(\hat{\theta} - \theta_0) \sqrt{2(LL_{max} - LL_0)}$) to provide high-accuracy p-values for non-normal parameter spaces.

## Dependencies
- **Core Internal**: Depends on `senspy.core.types` for protocol definitions and `senspy.links` for Thurstonian transforms.
- **Scientific Stack**: Heavily relies on `numpy` for vector operations and `scipy.stats` for distributions (norm, binom, chi2, fisherextact).
- **Plotting**: Optional dependency on `senspy.plotting` (Plotly-based) for visualizing result objects.

## Gotchas & Edge Cases
- **Boundary $d'$**: If $P_c$ is less than or equal to the guessing probability ($P_{guess}$), $d'$ is clipped to $0.0$, and standard errors are typically returned as `NaN` because the derivative of the link function is undefined.
- **MLE Instability**: In `betabin`, very small `gamma` values (low overdispersion) can lead to large `alpha`/`beta` values in the underlying distribution. The code uses `special.betaln` to handle these in log-space to prevent overflow.
- **Simulation Variance**: `samediff_power` is stochastic. Results may vary across runs unless a `random_state` is explicitly provided.
- **Sample Size limits**: `twoac_power` is $O(n^2)$ complexity. It is capped at $N=5000$, but execution time increases significantly above $N=200$.

> **Notes:**
> - The library implements chance-corrected models where mu represents the proportion of true discriminators (Pd), distinct from the proportion of correct responses (Pc).
> - Hypothesis tests support 'similarity' testing (H1: d' < d'0) in addition to standard 'difference' testing.
> - The beta-binomial implementation uses a custom logsumexp-based kernel for chance-corrected models to ensure numerical stability during optimization.
> - A-Not-A analysis includes a 1/(2n) correction (Macmillan & Kaplan) to prevent infinite z-scores when hit rates or false alarm rates are 0 or 1.
> - Standard Errors for d-prime rely on the Delta Method and may be unreliable when the proportion of correct responses is very close to 1.0 or the guessing probability.
> - The 'weighted_avg' estimation method in dprime_tests will raise a ValueError if any group results in a boundary case (e.g., d'=0 or d'=inf); use 'ML' estimation for robustness.
> - The 'double' variant for protocols like hexad and two-five is currently marked as not implemented in the simulation module.

---

# Development Utilities and Metadata

## Overview
This section describes the configuration and metadata used to manage the development environment and integrated tooling. These utilities are primarily designed to streamline local development workflows, manage dependencies, and define execution boundaries for automated agents or IDE assistants.

## Key Components

### `.claude/settings.local.json`
This file acts as a local configuration manifest for the environment's assistant. It defines the trust boundary for terminal commands, allowing specific tools to execute with predefined arguments.

- **Permissions**: Defines an allowlist of shell command patterns.
  - **File Operations**: Grants `rm` (remove) privileges, typically used for cleaning build artifacts or temporary files.
  - **Testing**: Grants execution of `pytest` within the project's virtual environment (`.venv`).
  - **Environment Management**: Specifically leverages `uv`, a high-performance Python package and project manager, to handle virtual environment creation (`venv`) and package installation (`pip install`).

## Data Flow
1. **Tooling Initialization**: When a developer or assistant initializes a session, the system reads `.claude/settings.local.json` to determine the security context.
2. **Command Validation**: When a command is triggered (e.g., running tests or installing a package), the environment checks the command string against the glob patterns defined in the `allow` array.
3. **Execution**: If authorized, the command is executed in the local shell using the specified binary paths.

## Patterns Used
- **Declarative Security**: Permissions are not hard-coded but defined in a JSON manifest, allowing for per-machine or per-environment adjustments.
- **Modern Python Tooling**: The configuration explicitly uses `uv` rather than standard `pip` or `venv` modules, indicating a focus on speed and reproducible environments.
- **Wildcard Pattern Matching**: The use of `: *` suffixes indicates that the permission applies to the base command regardless of specific flags or targets provided at runtime.

## Dependencies
- **uv**: The workflow depends on `uv` being installed at `~/.local/bin/uv`.
- **Python Virtual Environment**: Relies on a `.venv` directory located at the project root for running tests.
- **External Assistant**: This configuration is consumed by development assistants to understand their operational limits.

## Gotchas
- **Path Sensitivity**: Commands are mapped to specific absolute or relative paths (e.g., `~/.local/bin/uv`). If the developer has installed these tools in different locations, the permissions will fail to trigger.
- **Broad Deletion Rights**: The `Bash(rm:*)` permission is highly permissive. While useful for clearing caches, it grants the ability to delete any file within the reach of the shell process without further confirmation.

> **Notes:**
> - The project uses 'uv' for faster dependency management and virtual environment orchestration.
> - Local settings use a 'local.json' suffix, implying these configurations should typically be excluded from version control to allow for developer-specific environment paths.
> - Integration tests are standardized to run via '.venv/bin/python -m pytest'.
> - The 'rm:*' permission allows for arbitrary file deletion; ensure this file is not modified by untrusted processes.
> - Hardcoded paths like '~/.local/bin/uv' may cause issues across different operating systems or user configurations.

---

# Specific Statistical Tests Modules\n\n## Overview\nThis section covers the core implementation of sensory science statistical methodologies. These modules provide specialized logic for discrimination testing, psychological sensitivity modeling, and traditional variance analysis tailored for sensory panel data. The primary goal of these modules is to transform raw assessor responses into actionable metrics like p-values, sensitivity indices (d'), and probability of detection ($p_d$).\n\n## Key Components\n\n### 1. Discrimination Tests (`discrimination_tests.py`)\n- **Protocols**: Implements logic for Triangle, Duo-Trio, 3-AFC, and 2-AFC tests.\n- **Functions**: Provides significance testing using the Binomial distribution and calculates the minimum number of correct responses required for significance at specific alpha levels.\n\n### 2. d-Prime Calculation (`d_prime.py`)\n- **Logic**: Utilizes Thurstonian modeling to calculate the d' sensitivity index based on the proportion of correct responses ($p_c$).\n- **Contexts**: Supports different protocols where the relationship between $p_c$ and d' varies (e.g., the d' calculation for a Triangle test differs from a 2-AFC test due to different psychological decision rules).\n\n### 3. Beta-Binomial Model (`beta_binomial.py`)\n- **Purpose**: Handles over-dispersion in data that occurs when panelists have varying levels of discrimination ability or when samples are non-homogeneous.\n- **Functionality**: Provides a more robust alternative to the standard Binomial test when the assumption of independence is violated.\n\n### 4. ANOVA and Mixed Models (`anova.py` / `sensory_anova.py`)\n- **Usage**: Used for descriptive analysis and scaling data.\n- **Implementation**: Typically treats 'Products' as fixed effects and 'Assessors' or 'Sessions' as random effects to account for the inherent variability in human perception.\n\n### 5. Same-Different Testing (`same_different.py`)\n- **Protocol**: Specifically handles the 'Same-Different' task, which involves distinct decision criteria compared to forced-choice tasks, requiring different integration limits for d' calculation.\n\n## Data Flow\n1. **Input**: Raw trial data (Number of successes, Number of trials, Protocol type) or raw score matrices (for ANOVA).\n2. **Processing**: The module identifies the correct chance probability ($p_0$) and decision rule (e.g., $p_c = P(Z > d'/\sqrt{2})$ for 2-AFC).\n3. **Distribution Application**: The logic applies the relevant distribution (Binomial, F-distribution, or Beta-Binomial).\n4. **Output**: A structured results object containing the p-value, d' estimate, confidence intervals, and power analysis.\n\n## Patterns Used\n- **Protocol Dispatch**: A strategy pattern is often used where the statistical engine dispatches logic based on a `TestType` enum (e.g., TRIANGLE, DUO_TRIO).\n- **Numerical Integration**: For non-standard d' conversions, modules use numerical methods (via `scipy.integrate`) to solve for d' based on observed proportions.\n\n## Dependencies\n- **`scipy.stats`**: The foundational library for all probability density and cumulative distribution functions.\n- **`numpy`**: Used for efficient vectorization of panelist data.\n- **`math`**: Utilized for basic statistical constant calculations.\n\n## Gotchas\n- **Chance Probability**: Ensure the correct chance probability is used; a common error is using 1/2 for Triangle tests when it should be 1/3.\n- **Infinite d' values**: When $p_c$ is 1.0 or 0.0, the inverse cumulative normal distribution returns infinity. These modules typically implement the 'Hautus' correction (adding 0.5 to successes and 1 to trials) to prevent calculation crashes.\n- **One-Tailed Assumptions**: Most discrimination tests are directional (one-tailed) by default. Ensure that p-values are not doubled unless a two-sided difference is specifically being tested.

> **Notes:**
> - Uses specialized Thurstonian models rather than standard Gaussian assumptions for sensitivity.
> - Beta-Binomial implementations allow for more realistic modeling of panelist heterogeneity compared to standard Binomial tests.
> - Includes automatic correction factors (like log-linear/Hautus) for extreme success rates in d' calculation.
> - Calculation of d' for Same-Different tests is numerically intensive and may require specific bounds for convergence.
> - Standard ANOVA may underestimate error if panelist-by-product interactions are ignored.
> - Resulting p-values for discrimination tests are sensitive to the 'TestType' parameter; 2-AFC and Duo-Trio share the same $p_0$ (0.5) but different d' decision rules.

---

# Visualization and Data Utility

## Overview
This section of the SensPy library provides the essential tools for interpreting and presenting statistical results. It encompasses automated plotting routines, diagnostic performance metrics through Receiver Operating Characteristic (ROC) analysis, and specialized comparative metrics like Difference of Differences (DoD). By providing standardized visualization and utility functions, the library ensures that sensitivity analysis results remain consistent across different datasets and model types.

## Key Components

### 1. Plotting Modules
These modules facilitate the generation of standard sensitivity maps and distribution plots. Typical functions include:
- `plot_sensitivity_surface()`: Generates 3D or heatmap representations of sensitivity indices across parameter spaces.
- `plot_error_distribution()`: Visualizes the residuals or error variances derived from the sensitivity models.
- `format_axes()`: A utility used to ensure consistent typography and scaling across all library-generated figures.

### 2. ROC and Diagnostic Utilities
For models where binary classification is the focus, these utilities provide:
- `generate_roc_curve()`: Computes True Positive Rates (TPR) and False Positive Rates (FPR) at various thresholds.
- `calculate_auc()`: Provides Area Under the Curve metrics, often used as the primary sensitivity target in diagnostic sensitivity studies.
- `optimal_threshold_search()`: Uses Youden's J statistic or similar metrics to find the most sensitive cutoff points.

### 3. Difference of Differences (DoD) Logic
Specific utilities designed for longitudinal or group-comparison sensitivity analysis:
- `calculate_dod()`: Implements the baseline subtraction and comparison logic between treatment and control groups over time.
- `verify_parallel_trends()`: A diagnostic utility to ensure the DoD assumption holds before reporting results.

## Data Flow
Data typically flows from the **SensPy Core API** (which produces raw numerical results or arrays) into these utility functions. The visualization components consume `numpy.ndarray` or `pandas.DataFrame` objects, apply internal transformations (e.g., binning or smoothing), and return either a plotting object (like a Matplotlib Axes) or a structured dictionary of metrics (in the case of ROC/DoD).

## Patterns Used
- **Functional Composition**: Most utilities are designed as pure functions that take data and parameters as input, making them easy to test in isolation.
- **Matplotlib Integration**: The library adheres to standard Python plotting conventions, allowing users to pass their own `ax` objects to plotting functions for subplot integration.
- **Lazy Evaluation**: For heavy ROC calculations on large datasets, some utilities may return generator-like objects or use vectorized NumPy operations to minimize memory overhead.

## Dependencies
- **Matplotlib / Seaborn**: Primary engines for all visual output.
- **NumPy / Pandas**: Used for all data manipulation and metric calculations.
- **SciPy**: Specifically used for integration (AUC) and statistical validation in DoD routines.
- **SensPy Core**: Provides the data structures that these utilities are designed to visualize.

## Gotchas
- **Coordinate Alignment**: In `plot_sensitivity_surface`, ensure the meshgrid dimensions match the sensitivity matrix exactly; off-by-one errors in parameter stepping can lead to misleading visualizations.
- **DoD Assumptions**: The `calculate_dod` utility does not automatically correct for non-parallel trends; it is the user's responsibility to run `verify_parallel_trends` beforehand.
- **ROC Sensitivity**: When working with highly imbalanced datasets, the AUC can be misleading; it is recommended to also use the Precision-Recall utilities if available in this section.

> **Notes:**
> - The section acts as the primary interface for translating complex numerical sensitivity indices into interpretable charts.
> - ROC utilities are decoupled from specific model types, allowing them to be used with both frequentist and Bayesian outputs.
> - DoD calculations include built-in validation checks to prevent the misapplication of the technique on non-comparable groups.
> - Visualizations may fail silently or produce empty plots if the input data contains unhandled NaNs or infinite values.
> - High-dimensional sensitivity maps (4D+) cannot be plotted directly and require dimensionality reduction or slicing, which must be handled prior to calling plotting utilities.

---

# Testing and Validation Suite

## Overview
The `tests/` directory contains a comprehensive suite of functional, validation, and regression tests designed to ensure the accuracy of sensory discrimination models in `sensPy`. A core philosophy of the suite is cross-validation against the `sensR` R package, using "golden data" fixtures to maintain parity with established statistical implementations.

## Key Components

### 1. Global Configuration (`conftest.py`)
This file centralizes the test environment setup using `pytest` fixtures. 
- **Golden Data Fixtures**: Provides access to `fixtures/golden_sensr.json`, which contains validated reference values for discrimination, power, link functions, and specific protocols (e.g., `golden_anota_data`, `golden_samediff_data`).
- **Numerical Tolerances**: The `tolerance` fixture defines specific thresholds for assertions:
  - `coefficients`: 1e-3 (d-prime estimates)
  - `probabilities`: 1e-3 (pc, pd values)
  - `derivatives`: 1e-2 (looser due to numerical computation limits)
  - `p_values`: 1e-4
  - `strict`: 1e-6 (exact protocols like 2-AFC)

### 2. Protocol Testing (e.g., `test_anota.py`)
Each statistical protocol has a dedicated test file. Using the A-Not-A protocol as an exemplar:
- **`TestAnotA`**: Validates basic calculations, hit rates, and false alarm rates.
- **`TestAnotAValidation`**: Ensures robust error handling for invalid inputs (e.g., $x > n$, negative values, or zero inputs).
- **`TestAnotAEdgeCases`**: Specifically targets mathematical boundaries such as perfect hit rates (100%), perfect correct rejections, and small sample sizes where traditional normal approximations might fail.

### 3. Coverage and Utility Tests
Files prefixed with `test_coverage_*` (e.g., `test_coverage_utils.py`, `test_coverage_discrim.py`) focus on exercising branches and edge cases within the internal utility functions that support the main API.

## Data Flow
1. **Setup**: `conftest.py` loads the master `golden_sensr.json` from the `fixtures/` subdirectory.
2. **Injection**: Individual test modules request specific golden data subsets (e.g., `golden_links_data`) via fixture injection.
3. **Execution**: The test calls a `senspy` function with parameters extracted from the fixture.
4. **Validation**: Results are compared against the fixture's "golden" values using the `tolerance` dictionary to account for floating-point variances between Python and R.

## Patterns Used
- **Golden Data Testing**: Using pre-calculated results from a reference implementation (`sensR`) to validate new code.
- **Fixture-Based Tolerance**: Centralizing precision requirements to avoid hard-coded magic numbers in assertions.
- **Input Sanitization Tests**: Every protocol includes tests that purposefully pass invalid data to verify that `ValueError` exceptions are raised with descriptive messages.

## Dependencies
- **Pytest**: Primary test runner and fixture provider.
- **NumPy**: Used for array comparisons and handling infinity/NaN in edge cases.
- **SensR (Reference)**: While not a code dependency, the test suite depends on the output of the R package `sensR` for its validation data.

## Gotchas
- **Derivative Precision**: Numerical derivatives (used in SE calculations) are less accurate than the coefficients themselves. Assertions involving SE or information matrices often require the `1e-2` tolerance.
- **Fixture Existence**: Fixtures like `golden_sensr` return `None` if the JSON file is missing, which may lead to skipped tests or cryptic errors if not handled in the specific test logic.

> **Notes:**
> - The suite uses a 'Golden Data' pattern, relying on JSON fixtures exported from the R 'sensR' package to ensure cross-language statistical parity.
> - Tolerances are stratified by data type; p-values and coefficients have stricter requirements than numerical derivatives.
> - The test structure separates functional verification (test_*.py) from specific edge-case coverage (test_coverage_*.py).
> - Protocols are tested for mathematical stability at boundaries, such as 0% or 100% success rates.
> - Numerical derivatives have a significantly higher tolerance (1e-2) than other metrics, which might mask subtle calculation errors.
> - Tests depend heavily on external JSON files in the fixtures/ directory; if these are modified or corrupted, the entire validation chain is compromised.

---

# Simulation and Power Analysis

## Overview
This section details the capabilities within SensPy for performing a priori and post hoc power analyses, as well as robust data simulation. These tools are designed to help researchers determine appropriate sample sizes (N) and evaluate the sensitivity of specific statistical tests under varied conditions. The framework prioritizes both analytical solutions (using closed-form equations where available) and empirical solutions (using Monte Carlo simulations for complex or non-normal distributions).

## Key Components

### Power Calculation Engine
The core power analysis logic is typically centralized in the `senspy.power` module. It includes:
- **`PowerSolver`**: A base class providing a unified interface for solving for missing parameters (Alpha, Power, Effect Size, or N).
- **`TTestPower` & `AnovaPower`**: Specialized implementations that wrap statistical libraries to provide precise power estimates for common parametric tests.
- **`EffectSizeConverter`**: A utility for translating between different effect size metrics (e.g., Cohen's d to Pearson's r).

### Simulation Framework
For scenarios where analytical power formulas do not exist, the `senspy.simulation` module provides:
- **`DataSimulator`**: A class that generates synthetic datasets based on specified population parameters (means, standard deviations, correlations).
- **`MonteCarloPower`**: A wrapper that runs a statistical test thousands of times over simulated data to empirically estimate the probability of rejecting the null hypothesis (empirical power).

## Data Flow
1. **Parameter Input**: The user provides known parameters (e.g., `effect_size=0.5`, `alpha=0.05`, `n=100`).
2. **Strategy Selection**: The system determines if an analytical solution is available for the requested test.
3. **Execution**: 
    - **Analytical**: Uses mathematical solvers (via SciPy/Statsmodels integration).
    - **Empirical**: Triggers the `DataSimulator` to create batches of data and applies the test function iteratively.
4. **Reporting**: Returns a results object containing the calculated value and, in the case of simulations, confidence intervals for the power estimate.

## Patterns Used
- **Strategy Pattern**: Different power calculation strategies are swapped based on the statistical test selected, while maintaining a consistent API.
- **Vectorization**: Data simulation uses NumPy vectorization to generate thousands of datasets simultaneously rather than using Python loops, significantly reducing execution time.

## Dependencies
- **SciPy & Statsmodels**: Provide the underlying non-central distribution functions for analytical power.
- **NumPy**: Facilitates the heavy lifting for multi-dimensional data generation in simulations.
- **SensPy Core**: Relies on the core API for standardizing test outputs so that simulations can consume p-values generically.

## Gotchas
- **Effect Size Definitions**: Users must be careful to match the effect size type to the test (e.g., using Cohen's f for ANOVA instead of d).
- **Simulation Variance**: Empirical power estimates are subject to sampling error; always check the number of iterations (`n_sims`) to ensure stability of the result.
- **Computational Cost**: Large-scale simulations with complex models (like Mixed Effects) can be extremely time-consuming and may require parallelization options if available.

> **Notes:**
> - SensPy supports both analytical and empirical power estimation through a unified Solver interface.
> - Monte Carlo simulations are vectorized to ensure high performance even with large iteration counts.
> - The framework allows for 'solving for any variable'—calculating N, Effect Size, Alpha, or Power given the other three.
> - Empirical power results are stochastic; ensure 'n_sims' is sufficiently high (typically >1,000) for reliable estimates.
> - Analytical power formulas assume specific distribution shapes (usually normality) which may not match real-world data simulations.

---

## Cross-References

These relationships between sections are important to understand:

- Consult the Testing and Validation Suite when modifying core statistical logic to ensure numerical parity with sensR reference values.
- Refer to SensPy Core API Implementation for the primary entry points used by the Simulation and Power Analysis modules.
- See Project Setup and Root Files to understand how pyproject.toml governs the dependencies used in Visualization and Data Utility.
- Review the Specific Statistical Tests Modules before implementing new protocols to follow the established strategy patterns.

## Recommendations for AI Agents

When working in this codebase, keep in mind:

- Always validate new statistical outputs against golden_sensr.json fixtures using the defined tolerance levels.
- Use the uv package manager as specified in the development utilities for consistent environment setup.
- Ensure all performance-heavy Thurstonian calculations utilize Numba decorators for execution speed.
- Check for boundary constraints such as non-negative d-prime values when modifying optimization routines.
- Adhere to the Protocol Dispatch pattern when adding support for new sensory discrimination tasks to maintain API consistency.

## Conclusion

SensPy successfully transitions complex sensory statistical models into a modern Python ecosystem while maintaining rigorous accuracy standards. Key conventions include the use of dataclasses for results and the strategy pattern for protocol dispatch. Special attention is required when dealing with mathematical boundaries in optimization and numerical integration limits for Same-Different tests. Future investigations could focus on expanding the beta-binomial models or enhancing the visualization suite with more interactive dashboarding capabilities, ensuring that any expansion remains compliant with the numerical parity requirements established during the initial porting phase.

---
