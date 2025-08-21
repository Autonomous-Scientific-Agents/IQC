#!/usr/bin/env python3
"""
xyz_report_generator.py
----------------------
Generate a self‑contained interactive HTML report from an .xyz file.
• Uses RDKit (bond perception) to obtain SMILES / InChI and basic descriptors.
• Optionally fetches IUPAC names & more from PubChem.
• Embeds 3Dmol.js so structures are viewable in 3‑D right in the browser.

Usage
-----
$ python xyz_report_generator.py input.xyz  -o report.html  --pubchem
"""
import argparse, json, html, os, textwrap, sys
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Optional imports (handled gracefully if missing)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as Descriptors
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    Chem = None  # RDKit not available

try:
    import requests
except ImportError:
    requests = None  # PubChem fetch not available

# ---------------------------------------------------------------------------
PERIODIC = {
    "H": (1, 1.008),
    "He": (2, 4.0026),
    "Li": (3, 6.94),
    "Be": (4, 9.0122),
    "B": (5, 10.81),
    "C": (6, 12.011),
    "N": (7, 14.007),
    "O": (8, 15.999),
    "F": (9, 18.998),
    "Ne": (10, 20.18),
    "Na": (11, 22.99),
    "Mg": (12, 24.305),
    "Al": (13, 26.982),
    "Si": (14, 28.085),
    "P": (15, 30.974),
    "S": (16, 32.06),
    "Cl": (17, 35.45),
    "Ar": (18, 39.948),
    "K": (19, 39.098),
    "Ca": (20, 40.078),
    "Sc": (21, 44.956),
    "Ti": (22, 47.867),
    "V": (23, 50.942),
    "Cr": (24, 51.996),
    "Mn": (25, 54.938),
    "Fe": (26, 55.84),
    "Co": (27, 58.933),
    "Ni": (28, 58.693),
    "Cu": (29, 63.55),
    "Zn": (30, 65.4),
    "Ga": (31, 69.723),
    "Ge": (32, 72.63),
    "As": (33, 74.922),
    "Se": (34, 78.97),
    "Br": (35, 79.90),
    "Kr": (36, 83.80),
    "Rb": (37, 85.468),
    "Sr": (38, 87.62),
    "Y": (39, 88.906),
    "Zr": (40, 91.22),
    "Nb": (41, 92.906),
    "Mo": (42, 95.95),
    "Tc": (43, 96.906),
    "Ru": (44, 101.1),
    "Rh": (45, 102.906),
    "Pd": (46, 106.42),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_xyz(path: Path):
    """Return list of dicts: [{comment, atoms:[(elem,x,y,z)]}]"""
    lines = path.read_text().strip().splitlines()
    i = 0
    structs = []
    while i < len(lines):
        try:
            nat = int(lines[i].strip())
        except ValueError:
            break
        if i + nat + 1 >= len(lines):
            break
        comment = lines[i + 1].strip()
        atoms = []
        for l in lines[i + 2 : i + 2 + nat]:
            parts = l.split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        structs.append({"comment": comment, "atoms": atoms})
        i += nat + 2
    return structs


def xyz_block(struct):
    """Return an XYZ block (string) for 3Dmol or RDKit."""
    header = f"{len(struct['atoms'])}\n{struct['comment']}\n"
    body = "\n".join(f"{e} {x} {y} {z}" for e, x, y, z in struct["atoms"])
    return header + body


def rdkit_descriptors(block):
    if Chem is None:
        return {}
    try:
        mol = Chem.MolFromXYZBlock(block)
        rdDetermineBonds.DetermineConnectivity(mol)
        rdDetermineBonds.DetermineBondOrders(mol)
        if mol is None:
            return {}
        Chem.SanitizeMol(mol)
        
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        smiles = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)
        mw = Descriptors.CalcExactMolWt(mol)
        num_bonds = mol.GetNumBonds()

        return {"smiles": smiles, "inchi": inchi, "mw": mw, "formula": formula, "num_bonds": num_bonds}
    
    except Exception:
        return {}


def pubchem_lookup(smiles):
    if smiles is None or requests is None:
        return {}
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{requests.utils.quote(smiles)}/property/IUPACName,InChIKey/JSON"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        props = r.json()["PropertyTable"]["Properties"][0]
        return {"iupac": props.get("IUPACName"), "inchikey": props.get("InChIKey")}
    except Exception:
        return {}


def basic_counts(struct):
    cnt = Counter(e for e, *_ in struct["atoms"])
    formula = "".join(f"{el}{cnt[el] if cnt[el]>1 else ''}" for el in sorted(cnt))
    electrons = sum(PERIODIC.get(el, (0, 0))[0] * n for el, n in cnt.items())
    mass = sum(PERIODIC.get(el, (0, 0))[1] * n for el, n in cnt.items())
    return {"formula_est": formula, "mass_est": mass, "electrons": electrons}


# ---------------------------------------------------------------------------
# HTML template pieces (inline for simplicity)
# ---------------------------------------------------------------------------
VIEWER_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<script>
(function(){
  const structs = JSON.parse(document.getElementById('data-json').textContent);
  const container = document.getElementById('viewer-container');
  structs.forEach((s,idx)=>{
    const card=document.createElement('div');card.className='viewer-card';
    const vdiv=document.createElement('div');vdiv.className='viewer';card.appendChild(vdiv);
    const info=document.createElement('div');info.className='info';card.appendChild(info);
    container.appendChild(card);
    const viewer=$3Dmol.createViewer(vdiv,{backgroundColor:'white'});
    viewer.addModel(s.xyz,'xyz');viewer.setStyle({},{stick:{}});viewer.zoomTo();viewer.render();
    info.innerHTML=s.infoHTML;
  });
})();
</script>
"""

STYLE = """
<style>
body{font-family:system-ui,Arial,sans-serif;margin:0;padding:1rem;background:#f5f5f5;}
#viewer-container{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;}
.viewer-card{background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);padding:1rem;display:flex;flex-direction:column;}
.viewer{position:relative;width:100%;height:300px;border:1px solid #ccc;border-radius:4px;}
.info{margin-top:.5rem;font-size:.9rem;line-height:1.4;}
</style>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\"><title>XYZ Report</title>{style}</head>
<body>
<h1>XYZ Report – {fname}</h1>
<div id=\"viewer-container\"></div>
<script id=\"data-json\" type=\"application/json\">{payload}</script>
{viewer_js}
</body></html>"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(xyz_path: Path, output: Path, with_pubchem: bool):
    structs = parse_xyz(xyz_path)
    payload = []
    for s in structs:
        block = xyz_block(s)
        rd = rdkit_descriptors(block)
        counts = basic_counts(s)
        extra = pubchem_lookup(rd.get("smiles")) if with_pubchem else {}
        # info panel HTML
        info = textwrap.dedent(
            f"""
            <strong>Comment:</strong> {html.escape(s['comment'])}<br>
            <strong>Atoms:</strong> {len(s['atoms'])}<br>
            <strong>Number of Bonds:</strong> {rd.get('num_bonds','—')}<br>
            <strong>Formula (RDKit):</strong> {rd.get('formula') or '—'}<br>
            <strong>Formula (counted):</strong> {counts['formula_est']}<br>
            <strong>MW (RDKit):</strong> {rd.get('mw') or '—'}<br>
            <strong>MW (counted):</strong> {counts['mass_est']:.3f}<br>
            <strong>Electrons:</strong> {counts['electrons']}<br>
            <strong>SMILES:</strong> {rd.get('smiles','—')}<br>
            <strong>InChI:</strong> {rd.get('inchi','—')}<br>
            <strong>IUPAC:</strong> {extra.get('iupac','—')}<br>
        """
        ).strip()
        payload.append({"xyz": block, "infoHTML": info})

    html_out = HTML_TEMPLATE.format(
        fname=xyz_path.name,
        style=STYLE,
        payload=json.dumps(payload),
        viewer_js=VIEWER_JS,
    )
    output.write_text(html_out)
    print(f"Report written to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate HTML report from XYZ")
    ap.add_argument("xyz", type=Path, help="Input .xyz file")
    ap.add_argument("-o", "--output", type=Path, default=Path("report.html"))
    ap.add_argument(
        "--pubchem", action="store_true", help="Fetch IUPAC & InChIKey from PubChem"
    )
    args = ap.parse_args()
    build_report(args.xyz, args.output, args.pubchem)
