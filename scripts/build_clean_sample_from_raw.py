import sys
from pathlib import Path
try:
    import pandas as pd
except Exception:
    print('pandas is not available in the active Python environment. Install it with: conda install -n gtc-env pandas -y', file=sys.stderr)
    raise
raw = Path('LD2011_2014.txt')
if not raw.exists():
    raw = Path('LD2011_2014.txt.zip')
    if raw.exists():
        print('Found zip file; please unzip LD2011_2014.txt and place it in the repo root.', file=sys.stderr)
        sys.exit(2)
    print('Raw data file LD2011_2014.txt not found in repo root.', file=sys.stderr)
    sys.exit(2)
print(f'Reading first 100 rows from {raw} with sep=";" and decimal="," ...')
df = pd.read_csv('LD2011_2014.txt', sep=';', decimal=',', header=0, nrows=100)
df = df.iloc[:, 1:]
out = Path('uci_electricity_sample.cleaned.csv')
df.to_csv(out, index=False, header=False)
print(f'Wrote {out.resolve()}')
