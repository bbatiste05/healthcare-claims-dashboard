
# copilot/rag.py
import pandas as pd
from pathlib import Path

class SimpleRAG:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.icd = self._load_csv('data/external/icd10.csv', expected_cols=['code','system','short_title'])
        self.cpt = self._load_csv('data/external/cpt.csv', expected_cols=['code','system','short_title'])
        self.nppes = self._load_csv('data/external/nppes.csv', expected_cols=['npi','provider_name','taxonomy_specialty','state'])

    def _load_csv(self, rel, expected_cols=None):
        p = self.base / rel
        if p.exists():
            df = pd.read_csv(p)
            if expected_cols:
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"{rel} missing columns: {missing}")
            return df
        else:
            return pd.DataFrame()

    def search(self, query: str, k: int = 5):
        hits = []
        q = str(query).lower()
        for name, df in [('icd10', self.icd), ('cpt', self.cpt)]:
            for _, row in df.iterrows():
                hay = ' '.join(map(str, row.values)).lower()
                if any(tok in hay for tok in q.split()):
                    hits.append({"source": name, "row": row.to_dict()})
                    if len(hits) >= k:
                        break
        return hits

    def lookup_code(self, code: str):
        code = str(code).upper()
        r = {}
        if not self.icd.empty:
            match = self.icd[self.icd['code'].astype(str).str.upper() == code]
            if not match.empty:
                r['icd10'] = match.iloc[0].to_dict()
        if not self.cpt.empty:
            match = self.cpt[self.cpt['code'].astype(str).str.upper() == code]
            if not match.empty:
                r['cpt'] = match.iloc[0].to_dict()
        return r

    def lookup_npi(self, npi: str):
        if self.nppes.empty:
            return {}
        m = self.nppes[self.nppes['npi'].astype(str) == str(npi)]
        return m.iloc[0].to_dict() if not m.empty else {}
