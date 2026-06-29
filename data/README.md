cat > data/README.md << 'EOF'
# Data Sources

This repository does not track large raw data files due to GitHub size constraints.
Download and place them in the paths below before running any pipeline.

## Required files

### Geography
| File | Path | Source | Notes |
|------|------|--------|-------|
| `00mun.shp` (+ sidecar files) | `data/raw/geographic/municipalities/` | [INEGI - Marco Geoestadístico](https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463807469) | Municipal boundaries, EPSG:6372 |

### Security
| File | Path | Source | Notes |
|------|------|--------|-------|
| `Municipal-Delitos-2015-2025_abr2026.csv` | `data/raw/crime/` | [SESNSP](https://www.gob.mx/sesnsp/acciones-y-programas/datos-abiertos-de-incidencia-delictiva) | Municipal crime incidence 2015–2025, released April 2026 |

### Population
| File | Path | Source | Notes |
|------|------|--------|-------|
| `iter_00_cpv2010.csv` (+ sidecar files) | `data/raw/population/census_2010/` | [INEGI](https://www.inegi.org.mx/programas/ccpv/2010/) | National population census 2010 |
| `conjunto_de_datos_iter_00CSV20.csv` (+ sidecar files) | `data/raw/population/census_2020/` | [INEGI](https://www.inegi.org.mx/programas/ccpv/2020/) | National population census 2020 |

### Interim
| File | Path | Source | Notes |
|------|------|--------|-------|
| `gravity_matrix_2025.csv` | `data/interim/` | This document is produced by this project notebook gravity_matrix.ipynb | Spatial interaction matrix which uses population and distance |


## Notes
- All files should be placed exactly at the paths above; pipelines reference them by these locations.
- Coordinate system for shapefiles: EPSG:6372 (INEGI Lambert Conformal Conic) unless otherwise noted.
- The `data/processed/` directory is fully tracked by git.
EOF