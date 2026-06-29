import pandas as pd

def merge_thin_regimes(regime_dfs: dict, MIN_MUNI=30) -> dict:
    names = sorted(regime_dfs.keys())  # e.g. ['df_regime1', ..., 'df_regime10']

    merged = {}
    buffer = None  # holds a thin regime waiting to be absorbed

    for name in names:
        df = regime_dfs[name]
        n_munis = df['CVEGEO'].nunique() if 'CVEGEO' in df.columns else len(df)

        if buffer is not None:
            # Absorb the buffered thin regime into the current one
            df = pd.concat([buffer, df], ignore_index=True)
            buffer = None

        if n_munis < MIN_MUNI:
            print(f"  {name} has only {n_munis} municipalities — buffering to merge upward.")
            buffer = df  # hold it; merge into next regime
        else:
            merged[name] = df

    # If the last regime was thin, attach it to the last accepted regime
    if buffer is not None:
        if merged:
            last_key = sorted(merged.keys())[-1]
            merged[last_key] = pd.concat([merged[last_key], buffer], ignore_index=True)
            print(f"  Remaining thin regime merged into {last_key}.")
        else:
            merged['df_regime1'] = buffer

    # Rename keys to be sequential after merging
    final = {}
    for i, key in enumerate(sorted(merged.keys()), start=1):
        new_key = f"df_regime{i}"
        final[new_key] = merged[key]
        n = final[new_key]['CVEGEO'].nunique() if 'CVEGEO' in final[new_key].columns else len(final[new_key])
        print(f"  {new_key}: {n} unique municipalities")

    # Save post-merge regime assignments to crime_rates_regimes.csv
    regime_records = []
    for regime_name, regime_df in final.items():
        regime_num = int(regime_name.replace('df_regime', ''))
        temp = regime_df.copy()
        temp['regimes'] = regime_num
        regime_records.append(temp)

    regimes_df = pd.concat(regime_records, ignore_index=True)
    regimes_df['CVEGEO'] = regimes_df['CVEGEO'].astype(str).str.zfill(5)
    regimes_df.to_csv("data/interim/crime_rates_regimes.csv", index=False)
    print("  crime_rates_regimes.csv updated with post-merge regime assignments.")
    
    return final