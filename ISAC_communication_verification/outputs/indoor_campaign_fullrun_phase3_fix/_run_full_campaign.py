from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(r'D:\codex\ISAC_communication_verification')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualpol_rt import build_phase2_patterns_from_standard_files, run_indoor_campaign


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


need = [
    'LP_+45_new_6G7G_11pts.ffd',
    'LP_-45_new_6G7G_11pts.ffd',
    'RHCP_new_6G7G_11pts.ffd',
    'LHCP_new_6G7G_11pts.ffd',
]
root = Path(r'D:\OneDrive - postech.ac.kr')
found = {name: next(root.rglob(name)) for name in need}
ffd_dir = found['LP_+45_new_6G7G_11pts.ffd'].parent
outdir = Path(r'D:\codex\ISAC_communication_verification\outputs\indoor_campaign_fullrun_phase3_fix')

log(f'Using FFD dir: {ffd_dir}')
families = build_phase2_patterns_from_standard_files(ffd_dir)
log('Starting full indoor campaign with default settings')
result = run_indoor_campaign(
    families['tx_lp'],
    families['rx_lp'],
    families['tx_cp'],
    families['rx_cp'],
    output_dir=outdir,
)
log(f'Completed full run. Selected scenes: {sorted(result.full_compare_results.keys())}')
log(f'Artifacts: {result.artifact_paths}')
