from __future__ import annotations

import site
import sys
from pathlib import Path


def setup_local_vendor() -> None:
    project_root = Path(__file__).resolve().parents[1]

    project_paths = [
        project_root / "artifacts" / "manual_site",
        project_root / "artifacts" / "vendor",
        project_root / "artifacts" / "vendor_pkgs",
    ]
    user_site_paths: list[str] = []

    # Подключаем пользовательский site-packages, куда уже ставились lightgbm/xgboost.
    for user_site in site.getusersitepackages().split(";"):
        if user_site:
            user_site_paths.append(user_site)

    for candidate in project_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    for candidate_str in user_site_paths:
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
