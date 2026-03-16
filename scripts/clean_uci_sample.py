import sys
from pathlib import Path
try:
    import pandas as pd
except Exception as e:
    print('pandas is not available in the active Python environment. Install it with: conda install -n gtc-env pandas -y', file=sys.stderr)
    raise
p = Path('uci_electricity_sample.csv')
if not p.exists():
    print(f'Input file not found: {p.resolve()}', file=sys.stderr)
    sys.exit(2)
print(f'Reading {p} using sep=";" and decimal="," ...')
df = pd.read_csv(p, sep=';', decimal=',', header=0)
df = df.iloc[:, 1:]
out = Path('uci_electricity_sample.cleaned.csv')
df.to_csv(out, index=False, header=False)
print(f'Wrote {out.resolve()}')
