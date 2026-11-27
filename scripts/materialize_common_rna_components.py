"""Compatibility entry point for the patient RNA builder installed on 149."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    script = directory / 'materialize_patient_star_rna.py'
    if not script.is_file():
        script = directory / 'source_integration/repo/scripts/materialize_patient_star_rna.py'
    runpy.run_path(str(script), run_name='__main__')
