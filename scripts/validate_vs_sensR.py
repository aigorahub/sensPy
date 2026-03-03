"""
Validation script: sensPy vs sensR
Calls R via subprocess and compares results with Python sensPy.
"""
import subprocess
import sys
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────────

def run_r(code):
    """Run R code and return stdout."""
    full = f'library(sensR)\n{code}'
    result = subprocess.run(
        ['Rscript', '-e', full],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        err = result.stderr.strip().split('\n')[-1][:120]
        print(f"  R ERROR: {err}")
        return None
    return result.stdout.strip()


# Method name mapping: sensPy -> sensR
R_METHOD = {
    "triangle": "triangle",
    "twoafc": "twoAFC",
    "duotrio": "duotrio",
    "threeafc": "threeAFC",
    "tetrad": "tetrad",
    "hexad": "hexad",
    "twofive": "twofive",
}

R_STAT = {
    "exact": "exact",
    "likelihood": "likelihood",
    "wald": "Wald",
    "score": "score",
}

pass_count = 0
fail_count = 0
skip_count = 0


def compare(label, py_val, r_val, tol=1e-4):
    global pass_count, fail_count, skip_count
    r_str = str(r_val).strip()
    if r_str in ('NA', 'NaN', 'Inf', '-Inf', ''):
        skip_count += 1
        return
    try:
        py_f = float(py_val)
        r_f = float(r_str)
    except (ValueError, TypeError):
        skip_count += 1
        return
    diff = abs(py_f - r_f)
    ok = diff < tol
    tag = 'PASS' if ok else 'FAIL'
    if ok:
        pass_count += 1
        print(f"  {tag} | {label:45s} | Py={py_f:12.6f} | R={r_f:12.6f}")
    else:
        fail_count += 1
        print(f"  {tag} | {label:45s} | Py={py_f:12.6f} | R={r_f:12.6f} | d={diff:.2e}")


# ── Imports ──────────────────────────────────────────────────────────────────

from senspy import (
    discrim, betabin, twoac, samediff, anota, dod,
    discrim_power, dprime_power, discrim_sample_size, dprime_sample_size,
    dprime_test, dprime_compare, dprime_table,
    rescale, psy_fun, psy_inv, psy_deriv,
    pc_to_pd, pd_to_pc, auc, sdt,
    optimal_tau, par2prob_dod,
)

# ═════════════════════════════════════════════════════════════════════════════
print("=" * 95)
print("1. discrim() — Basic discrimination analysis")
print("=" * 95)

test_cases = [
    (80, 100, "triangle", "exact"),
    (60, 100, "duotrio", "exact"),
    (75, 100, "twoafc", "likelihood"),
    (50, 100, "threeafc", "wald"),
    (45, 100, "tetrad", "score"),
    (30, 100, "twoafc", "exact"),
    (90, 120, "triangle", "likelihood"),
]

for correct, total, method, stat in test_cases:
    rm = R_METHOD[method]
    rs = R_STAT[stat]
    print(f"\n--- {correct}/{total}, {method}, {stat} ---")

    py = discrim(correct, total, method=method, statistic=stat)

    r_out = run_r(f'''
        r <- discrim({correct}, {total}, method="{rm}", statistic="{rs}")
        cat(r$coefficients["d-prime", "Estimate"],
            r$coefficients["d-prime", "Std. Error"],
            r$coefficients["pc", "Estimate"],
            r$coefficients["pd", "Estimate"],
            r$p.value, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("d_prime", py.d_prime, vals[0])
        compare("se_d_prime", py.se_d_prime, vals[1])
        compare("pc", py.pc, vals[2])
        compare("pd", py.pd, vals[3])
        compare("p_value", py.p_value, vals[4])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("2. discrim() — Confidence Intervals")
print("=" * 95)

ci_cases = [
    (80, 100, "triangle", "exact"),
    (60, 100, "twoafc", "likelihood"),
    (70, 100, "duotrio", "exact"),
]

for correct, total, method, stat in ci_cases:
    rm = R_METHOD[method]
    rs = R_STAT[stat]
    print(f"\n--- {method}, {stat}, {correct}/{total} ---")
    py = discrim(correct, total, method=method, statistic=stat)

    for param in ["d_prime", "pc", "pd"]:
        r_row = {"d_prime": "d-prime", "pc": "pc", "pd": "pd"}[param]
        py_ci = py.confint(parameter=param)
        r_out = run_r(f'''
            r <- discrim({correct}, {total}, method="{rm}", statistic="{rs}")
            ci <- confint(r)
            cat(ci["{r_row}", "Lower"], ci["{r_row}", "Upper"], sep="\\n")
        ''')
        if r_out:
            vals = r_out.split('\n')
            compare(f"CI lower ({param})", py_ci[0], vals[0])
            compare(f"CI upper ({param})", py_ci[1], vals[1])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("3. discrim() — Similarity Tests")
print("=" * 95)

sim_cases = [
    (40, 100, "triangle", 1.0, "exact"),
    (55, 100, "twoafc", 0.5, "likelihood"),
]

for correct, total, method, dp0, stat in sim_cases:
    rm = R_METHOD[method]
    rs = R_STAT[stat]
    print(f"\n--- {method}, d_prime0={dp0}, {stat} ---")
    py = discrim(correct, total, method=method, d_prime0=dp0,
                 statistic=stat, test="similarity")
    r_out = run_r(f'''
        r <- discrim({correct}, {total}, method="{rm}", d.prime0={dp0},
                     statistic="{rs}", test="similarity")
        cat(r$coefficients["d-prime", "Estimate"], r$p.value, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("d_prime", py.d_prime, vals[0])
        compare("p_value", py.p_value, vals[1])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("4. rescale() — d' -> pc, pd")
print("=" * 95)

for method in ["triangle", "twoafc", "duotrio", "threeafc", "tetrad"]:
    rm = R_METHOD[method]
    for dp in [0.5, 1.0, 2.0, 3.0]:
        py_res = rescale(d_prime=dp, method=method)
        r_out = run_r(f'''
            r <- rescale(d.prime={dp}, method="{rm}")
            co <- coef(r)
            cat(co$pc, co$pd, sep="\\n")
        ''')
        if r_out:
            vals = r_out.split('\n')
            compare(f"{method} d'={dp} -> pc", py_res.pc, vals[0])
            compare(f"{method} d'={dp} -> pd", py_res.pd, vals[1])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("5. psyfun / psyinv / psyderiv")
print("=" * 95)

for method in ["triangle", "twoafc", "duotrio", "threeafc", "tetrad"]:
    rm = R_METHOD[method]
    print(f"\n--- {method} ---")
    for dp in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        py_pc = float(psy_fun(dp, method=method))
        r_out = run_r(f'cat(psyfun({dp}, method="{rm}"))')
        if r_out:
            compare(f"psyfun d'={dp}", py_pc, r_out)

    for pc_val in [0.5, 0.7, 0.9]:
        py_dp = float(psy_inv(pc_val, method=method))
        r_out = run_r(f'cat(psyinv({pc_val}, method="{rm}"))')
        if r_out:
            compare(f"psyinv pc={pc_val}", py_dp, r_out)

    for dp in [0.5, 1.0, 2.0]:
        py_d = float(psy_deriv(dp, method=method))
        r_out = run_r(f'cat(psyderiv({dp}, method="{rm}"))')
        if r_out:
            compare(f"psyderiv d'={dp}", py_d, r_out)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("6. betabin() — Beta-Binomial model")
print("=" * 95)

bb_data = "3,10,5,10,7,10,4,10,6,10,8,10,2,10,5,10,9,10,4,10"

for method in ["duotrio", "triangle", "twoafc"]:
    rm = R_METHOD[method]
    for corrected in [True, False]:
        corr_r = "TRUE" if corrected else "FALSE"
        print(f"\n--- {method}, corrected={corrected} ---")

        data = np.array([
            [3, 10], [5, 10], [7, 10], [4, 10], [6, 10],
            [8, 10], [2, 10], [5, 10], [9, 10], [4, 10],
        ])
        py = betabin(data, method=method, corrected=corrected)

        r_out = run_r(f'''
            d <- matrix(c({bb_data}), nrow=10, ncol=2, byrow=TRUE)
            r <- betabin(d, method="{rm}", corrected={corr_r})
            co <- coef(r)
            cat(co[1], co[2], r$logLik, sep="\\n")
        ''')
        if r_out:
            vals = r_out.split('\n')
            compare("mu", py.mu, vals[0])
            compare("gamma", py.gamma, vals[1])
            compare("log_likelihood", py.log_likelihood, vals[2])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("7. twoac() — 2-AC Protocol")
print("=" * 95)

twoac_cases = [
    [40, 20, 60],
    [30, 40, 50],
    [10, 80, 30],
    [55, 10, 35],
]

for d in twoac_cases:
    print(f"\n--- data={d} ---")
    py = twoac(d, statistic="likelihood")
    r_out = run_r(f'''
        r <- twoAC(c({d[0]},{d[1]},{d[2]}), statistic="likelihood")
        cat(r$coefficients[1,1], r$coefficients[2,1],
            r$coefficients[1,2], r$coefficients[2,2],
            r$logLik, r$p.value, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("tau", py.tau, vals[0])
        compare("d_prime", py.d_prime, vals[1])
        compare("se_tau", py.se_tau, vals[2])
        compare("se_d_prime", py.se_d_prime, vals[3])
        compare("log_likelihood", py.log_likelihood, vals[4])
        if py.p_value is not None and len(vals) > 5:
            compare("p_value", py.p_value, vals[5])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("8. samediff() — Same-Different Protocol")
print("=" * 95)

sd_cases = [
    (45, 5, 20, 30),
    (80, 20, 10, 90),
    (30, 10, 15, 25),
]

for ss, ds, sd, dd in sd_cases:
    print(f"\n--- ss={ss}, ds={ds}, sd={sd}, dd={dd} ---")
    py = samediff(nsamesame=ss, ndiffsame=ds, nsamediff=sd, ndiffdiff=dd)
    r_out = run_r(f'''
        r <- samediff(nsamesame={ss}, ndiffsame={ds},
                      nsamediff={sd}, ndiffdiff={dd})
        cat(r$coef["tau"], r$coef["delta"],
            r$se["tau"], r$se["delta"],
            r$logLik, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("tau", py.tau, vals[0])
        compare("delta", py.delta, vals[1])
        compare("se_tau", py.se_tau, vals[2])
        compare("se_delta", py.se_delta, vals[3])
        compare("log_likelihood", py.log_likelihood, vals[4])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("9. anota() — A-Not-A Protocol")
print("=" * 95)

anota_cases = [
    (80, 100, 90, 100),
    (60, 80, 70, 80),
    (40, 50, 45, 50),
]

# NOTE: sensPy x2 = correct rejections; R x2 = false alarms
# So R_x2 = n2 - sensPy_x2
for x1, n1, x2, n2 in anota_cases:
    r_x2 = n2 - x2  # convert: sensPy correct rejections -> R false alarms
    print(f"\n--- x1={x1}, n1={n1}, x2={x2}, n2={n2} (R_x2={r_x2}) ---")
    py = anota(x1, n1, x2, n2)
    r_out = run_r(f'''
        r <- AnotA({x1}, {n1}, {r_x2}, {n2})
        cat(r$coefficients["d-prime"], r$se["d-prime"], r$p.value, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("d_prime", py.d_prime, vals[0])
        compare("se_d_prime", py.se_d_prime, vals[1])
        compare("p_value", py.p_value, vals[2])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("10. dod() — Degree of Difference")
print("=" * 95)

dod_cases = [
    ([25, 22, 33, 20], [10, 15, 30, 45]),
    ([40, 30, 20, 10], [5, 15, 30, 50]),
]

for same, diff in dod_cases:
    s_str = ','.join(map(str, same))
    d_str = ','.join(map(str, diff))
    print(f"\n--- same={same}, diff={diff} ---")
    py = dod(same, diff)
    r_out = run_r(f'''
        r <- dod(c({s_str}), c({d_str}))
        tau_str <- paste(r$tau, collapse="\\n")
        se <- r$coefficients["d.prime", "Std. Error"]
        cat(r$d.prime, se, r$logLik, r$p.value, tau_str, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare("d_prime", py.d_prime, vals[0])
        compare("se_d_prime", py.se_d_prime, vals[1])
        compare("log_lik", py.log_lik, vals[2])
        compare("p_value", py.p_value, vals[3])
        for i, (py_t, r_t) in enumerate(zip(py.tau, vals[4:])):
            compare(f"tau[{i}]", py_t, r_t)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("11. optimal_tau() & par2prob_dod()")
print("=" * 95)

for dp in [0.5, 1.0, 2.0]:
    for ncat in [3, 4, 5]:
        py_res = optimal_tau(dp, ncat=ncat)
        py_tau = py_res['tau']
        r_out = run_r(f'''
            r <- optimal_tau({dp}, n.cat={ncat})
            cat(paste(r, collapse="\\n"))
        ''')
        if r_out:
            vals = r_out.split('\n')
            print(f"\n--- optimal_tau d'={dp}, n_cat={ncat} ---")
            for i in range(min(len(py_tau), len(vals))):
                compare(f"tau[{i}]", py_tau[i], vals[i])

print("\n--- par2prob_dod ---")
tau = np.array([1.0, 2.0, 3.0])
dp = 1.5
py_prob = par2prob_dod(tau, dp)
r_out = run_r(f'''
    r <- par2prob_dod(c(1.0, 2.0, 3.0), 1.5)
    cat(paste(as.vector(r), collapse="\\n"))
''')
if r_out:
    vals = r_out.split('\n')
    r_flat = [float(v) for v in vals]
    # R returns column-major (same row first, then diff row)
    n_cols = py_prob.shape[1]
    for i in range(py_prob.shape[0]):
        for j in range(py_prob.shape[1]):
            label = f"prob[{'same' if i==0 else 'diff'},{j}]"
            r_idx = i + j * 2  # column-major order
            if r_idx < len(r_flat):
                compare(label, py_prob[i, j], r_flat[r_idx])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("12. discrim_power() / dprime_power()")
print("=" * 95)

power_cases = [
    (0.5, 100, 1/3, "exact"),
    (0.3, 50, 0.5, "exact"),
    (0.4, 80, 1/3, "normal"),
]

for pd_a, n, pg, stat in power_cases:
    py_pow = discrim_power(pd_a=pd_a, sample_size=n, p_guess=pg, statistic=stat)
    r_out = run_r(f'cat(discrimPwr({pd_a}, {n}, pGuess={pg}, statistic="{stat}"))')
    if r_out:
        compare(f"power pd_a={pd_a}, n={n}", py_pow, r_out)

print("\n--- dprime_power ---")
dp_cases = [
    (1.0, 100, "triangle", "exact"),
    (1.5, 50, "twoafc", "exact"),
    (2.0, 80, "duotrio", "exact"),
]

for dpa, n, method, stat in dp_cases:
    rm = R_METHOD[method]
    py_pow = dprime_power(d_prime_a=dpa, sample_size=n, method=method, statistic=stat)
    py_pd = rescale(d_prime=dpa, method=method).pd
    pg_r = run_r(f'cat(psyfun(0, method="{rm}"))')
    if pg_r:
        r_out = run_r(f'cat(discrimPwr({float(py_pd)}, {n}, pGuess={pg_r}, statistic="{stat}"))')
        if r_out:
            compare(f"d'={dpa}, {method}, n={n}", py_pow, r_out)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("13. discrim_sample_size()")
print("=" * 95)

ss_cases = [
    (0.5, 0.8, 1/3, "exact"),
    (0.3, 0.9, 0.5, "exact"),
    (0.4, 0.8, 1/3, "normal"),
]

for pd_a, power, pg, stat in ss_cases:
    py_n = discrim_sample_size(pd_a=pd_a, target_power=power, p_guess=pg, statistic=stat)
    r_out = run_r(f'cat(discrimSS({pd_a}, {power}, pGuess={pg}, statistic="{stat}"))')
    if r_out:
        compare(f"n: pd_a={pd_a}, power={power}", py_n, r_out)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("14. pc_to_pd() / pd_to_pc()")
print("=" * 95)

for pg in [1/3, 0.5, 0.1]:
    for pc_val in [0.5, 0.7, 0.9]:
        if pc_val >= pg:
            py_pd = float(pc_to_pd(pc_val, pg))
            r_out = run_r(f'cat(pc2pd({pc_val}, {pg}))')
            if r_out:
                compare(f"pc2pd pc={pc_val}, pg={pg:.2f}", py_pd, r_out)

    for pd_val in [0.0, 0.3, 0.5, 1.0]:
        py_pc = float(pd_to_pc(pd_val, pg))
        r_out = run_r(f'cat(pd2pc({pd_val}, {pg}))')
        if r_out:
            compare(f"pd2pc pd={pd_val}, pg={pg:.2f}", py_pc, r_out)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("15. AUC()")
print("=" * 95)

for dp in [0.5, 1.0, 2.0, 3.0]:
    py_auc = auc(dp)
    r_out = run_r(f'cat(AUC({dp}))')
    if r_out:
        compare(f"AUC d'={dp}", py_auc.value, r_out)

print("\n--- AUC with SE ---")
py_auc = auc(1.5, se_d=0.3)
r_out = run_r(f'''
    r <- AUC(1.5, se.d=0.3)
    cat(r$value, r$lower, r$upper, sep="\\n")
''')
if r_out:
    vals = r_out.split('\n')
    compare("AUC value", py_auc.value, vals[0])
    compare("AUC lower", py_auc.lower, vals[1])
    compare("AUC upper", py_auc.upper, vals[2])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("16. Cross-protocol comparison — discrim() 70/100")
print("=" * 95)

all_protocols = ["triangle", "duotrio", "twoafc", "threeafc", "tetrad"]
correct, total = 70, 100

for method in all_protocols:
    rm = R_METHOD[method]
    py = discrim(correct, total, method=method)
    r_out = run_r(f'''
        r <- discrim({correct}, {total}, method="{rm}")
        cat(r$coefficients["d-prime", "Estimate"],
            r$coefficients["d-prime", "Std. Error"],
            r$p.value, sep="\\n")
    ''')
    if r_out:
        vals = r_out.split('\n')
        compare(f"{method:12s} d_prime", py.d_prime, vals[0])
        compare(f"{method:12s} se", py.se_d_prime, vals[1])
        compare(f"{method:12s} p_value", py.p_value, vals[2])


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("VALIDATION SUMMARY")
print("=" * 95)
total_tests = pass_count + fail_count
print(f"  PASS: {pass_count}")
print(f"  FAIL: {fail_count}")
print(f"  SKIP: {skip_count}")
print(f"  Total compared: {total_tests}")
if total_tests > 0:
    print(f"  Pass rate: {pass_count/total_tests:.1%}")
print("=" * 95)

sys.exit(1 if fail_count > 0 else 0)
