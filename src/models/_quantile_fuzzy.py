import pandas as pd
import numpy as np


def _trapezoidal_mf(x, a, b, c, d):
    """
    Trapezoidal membership function on scalar x.
    Full membership [b, c], ramps on [a,b] and [c,d], zero outside.
    """
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # c < x < d
        return (d - x) / (d - c)


def _fuzzy_memberships(df_function, BINS=6):
    """
    Given a crime_rate value and the bin edges from pd.qcut,
    return a dict {state_label: membership} using trapezoidal MFs.

    MF for state k is centred on (bins[k-1], bins[k]) with shoulders
    extending halfway to the neighbouring bin centres.
    """
    n_states = len(BINS) - 1
    centres = [(BINS[k] + BINS[k + 1]) / 2.0 for k in range(n_states)]
    crime_rate = df_function

    memberships = {}
    for k in range(n_states):
        state_label = k + 1
        b = BINS[k]      # left hard edge
        c = BINS[k + 1]  # right hard edge

        # Shoulder to the left: halfway to previous centre (or -inf for first)
        a = centres[k - 1] if k > 0 else BINS[0] - 1e-9

        # Shoulder to the right: halfway to next centre (or +inf for last)
        d = centres[k + 1] if k < n_states - 1 else BINS[-1] + 1e-9

        memberships[state_label] = _trapezoidal_mf(crime_rate, a, b, c, d)

    # Normalise so memberships sum to 1 (avoids zero-vector at hard edges)
    total = sum(memberships.values())
    if total > 0:
        memberships = {s: v / total for s, v in memberships.items()}
    else:
        # Fallback: assign full membership to crisp state
        crisp = int(pd.cut([crime_rate], bins=BINS,
                           labels=list(range(1, n_states + 1)),
                           include_lowest=True)[0])
        memberships = {s: (1.0 if s == crisp else 0.0) for s in range(1, n_states + 1)}

    return memberships


def assign_quantiles(regime_dfs: dict, N_QUANTILES=5, BINS=5):

    result = {}
    for name, df in regime_dfs.items():
        df = df.copy()

        # First pass: discover how many unique bin edges survive after deduplication
        _, BINS = pd.qcut(
            df['crime_rate'],
            q=N_QUANTILES,
            retbins=True,
            duplicates='drop'
        )
        actual_n = len(BINS) - 1  # real number of bins after deduplication

        if actual_n < N_QUANTILES:
            print(f"  {name}: requested {N_QUANTILES} quantiles but only "
                  f"{actual_n} unique bin edges found using {actual_n} states.")

        # Second pass: crisp state assignment (anchor for state_means and printing)
        df['state'] = pd.qcut(
            df['crime_rate'],
            q=N_QUANTILES,
            labels=list(range(1, actual_n + 1)),
            duplicates='drop'
        ).astype(int)

        # Fuzzy membership vector per row
        df['membership'] = df['crime_rate'].apply(
            lambda r: _fuzzy_memberships(r, BINS)
        )

        # Mean violence rate per state used later for defuzzification
        state_means = (
            df.groupby('state')['crime_rate']
            .mean()
            .rename('state_mean')
        )
        df = df.join(state_means, on='state')

        actual_states = sorted(df['state'].unique())
        print(f"{name}: {len(actual_states)} states | bins: {np.round(BINS, 1)}")
        print(f"  State means: { {s: round(state_means[s], 1) for s in actual_states} }")
        result[name] = df

    return result