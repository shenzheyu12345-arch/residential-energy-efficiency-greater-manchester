"""Download official ONS 2021 LSOA boundaries for Greater Manchester only."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd


SERVICE_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
    "FeatureServer/0/query"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analytical-data",
        type=Path,
        default=root / "data" / "processed" / "lsoa_analytical_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "spatial" / "gm_lsoa_2021_bgc_v5.gpkg",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data" / "spatial" / "gm_lsoa_2021_bgc_v5_metadata.json",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_batch(codes: list[str]) -> gpd.GeoDataFrame:
    quoted = ",".join(f"'{code}'" for code in codes)
    command = [
        "curl.exe",
        "--http1.1",
        "--silent",
        "--show-error",
        "--fail",
        "--retry",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "120",
        "--data-urlencode",
        f"where=LSOA21CD IN ({quoted})",
        "--data-urlencode",
        "outFields=LSOA21CD,LSOA21NM,BNG_E,BNG_N,LAT,LONG",
        "--data-urlencode",
        "returnGeometry=true",
        "--data-urlencode",
        "outSR=4326",
        "--data-urlencode",
        "f=geojson",
        SERVICE_URL,
    ]
    response = None
    for attempt in range(1, 6):
        try:
            response = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=150,
            )
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt == 5:
                stderr = getattr(exc, "stderr", "")
                raise RuntimeError(
                    f"Boundary batch failed after {attempt} attempts: {stderr}"
                ) from exc
            delay = 2**attempt
            print(
                f"Transient boundary download failure; retrying in {delay}s "
                f"(attempt {attempt}/5)",
                flush=True,
            )
            time.sleep(delay)
    assert response is not None
    payload = json.loads(response.stdout)
    if "error" in payload:
        raise RuntimeError(f"ArcGIS query failed: {payload['error']}")
    return gpd.GeoDataFrame.from_features(payload.get("features", []), crs="EPSG:4326")


def main() -> None:
    args = parse_args()
    analytical = pd.read_csv(args.analytical_data, usecols=["lsoa21"], dtype="string")
    codes = sorted(analytical["lsoa21"].dropna().unique().tolist())
    if len(codes) != 1702:
        raise ValueError(f"Expected 1,702 analytical LSOAs; found {len(codes):,}")

    frames = []
    batches = list(chunks(codes, args.batch_size))
    for index, batch in enumerate(batches, start=1):
        frame = fetch_batch(batch)
        frames.append(frame)
        print(
            f"Downloaded boundary batch {index}/{len(batches)}: "
            f"requested={len(batch)}, returned={len(frame)}",
            flush=True,
        )
    boundaries = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )
    boundaries = boundaries.rename(columns={"LSOA21CD": "lsoa21", "LSOA21NM": "lsoa21_name"})
    boundaries = boundaries.drop_duplicates("lsoa21", keep="last")

    returned = set(boundaries["lsoa21"])
    missing = sorted(set(codes) - returned)
    extra = sorted(returned - set(codes))
    if missing or extra:
        raise ValueError(f"Boundary mismatch. Missing={missing[:10]}, extra={extra[:10]}")
    if boundaries.geometry.isna().any() or boundaries.geometry.is_empty.any():
        raise ValueError("Downloaded boundary data contains missing or empty geometries")

    invalid_before = int((~boundaries.geometry.is_valid).sum())
    if invalid_before:
        boundaries["geometry"] = boundaries.geometry.make_valid()
    invalid_after = int((~boundaries.geometry.is_valid).sum())
    if invalid_after:
        raise ValueError(f"{invalid_after} invalid geometries remain after make_valid")

    boundaries = boundaries.sort_values("lsoa21").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boundaries.to_file(args.output, layer="gm_lsoa_2021_bgc_v5", driver="GPKG")

    metadata = {
        "title": "Lower layer Super Output Areas (December 2021) Boundaries EW BGC (V5)",
        "publisher": "Office for National Statistics",
        "source_feature_service": SERVICE_URL.rsplit("/query", 1)[0],
        "source_item_id": "68515293204e43ca8ab56fa13ae8a547",
        "boundary_type": "BGC - Generalised (20m), clipped to the coastline",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(boundaries),
        "stored_crs": str(boundaries.crs),
        "invalid_geometries_before_repair": invalid_before,
        "invalid_geometries_after_repair": invalid_after,
        "licence": "Open Government Licence v3.0",
        "copyright": (
            "Source: Office for National Statistics licensed under the Open Government "
            "Licence v.3.0. Contains OS data Crown copyright and database right."
        ),
    }
    with args.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote boundaries: {args.output}")
    print(f"Wrote metadata: {args.metadata}")


if __name__ == "__main__":
    main()
