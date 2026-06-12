# hstha_contsubpipe_extended

A small, configuration-driven pipeline for first-pass HST narrowband continuum
subtraction of the PHANGS HST sample.

The current implementation keeps to a plain no-MUSE workflow: zero-to-NaN
preprocessing, narrowband padding removal, foreground extinction correction,
linear- or log-space continuum interpolation between F555W and F814W, propagated error
maps, conversion to integrated flux, and a configurable fixed [NII] correction.
It does not yet do anchoring, PSF matching, or reprojection.

## Directory Layout

- `config/paths.yaml` - root input and output directories.
- `config/files.yaml` - filename and glob templates.
- `config/galaxies.yaml` - galaxy list, defaults, and per-galaxy overrides.
- `src/hstha_contsubpipe_extended/pipeline/` - staged continuum-subtraction pipeline.
- `src/hstha_contsubpipe_extended/pipeline/cli.py` - `hstha-contsub` command-line entry point.
- `src/hstha_contsubpipe_extended/tasks/run_contsub.py` - task-registry wrapper.

Outputs are written to:

```text
/lustre/opsw/work/abarnes/phangs/HST_WORK/data_contsub
```

The main products are:

```text
{galaxy}_{narrow_filter}_contsub_flux_{linear|log}.fits
{galaxy}_{narrow_filter}_contsub_flux_err_{linear|log}.fits
{galaxy}_{narrow_filter}_continuum_flux_{linear|log}.fits
{galaxy}_{narrow_filter}_continuum_flux_err_{linear|log}.fits
{galaxy}_{narrow_filter}_halpha_flux_nii_corr_{linear|log}.fits
{galaxy}_{narrow_filter}_halpha_flux_nii_corr_err_{linear|log}.fits
contsub_manifest.csv
logs/contsub_run_{run_id}.log
logs/contsub_run_{run_id}.jsonl
```

Each run writes a timestamped human-readable audit log and a matching JSONL
audit stream. The logs include the selected stages, run timing, per-galaxy
settings, input/output paths, bandpass values and sources, derived weights, and
failure messages.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
pip install -e .
```

## Run The Pipeline

First do a dry run. This checks which files will be used without writing FITS
products:

```bash
hstha-contsub --dry-run
```

Run one galaxy:

```bash
hstha-contsub --galaxy ngc5068
```

Run multiple galaxies:

```bash
hstha-contsub --galaxy ngc5068 --galaxy ngc2903
```

Run the full configured sample:

```bash
hstha-contsub
```

Overwrite existing outputs:

```bash
hstha-contsub --galaxy ngc5068 --overwrite
```

The same workflow is also available through the task registry:

```bash
python -c "from hstha_contsubpipe_extended.tasks import run_task; run_task('run_contsub', galaxies=['ngc5068'], dry_run=True)"
```

## Pipeline Structure

The continuum-subtraction workflow is implemented as a small staged package
under `src/hstha_contsubpipe_extended/pipeline/`:

- `galaxy_config.py` loads the target list, defaults, and overrides.
- `discovery.py` resolves science and error FITS files from `files.yaml`.
- `fits_ops.py`, `bandpass.py`, `extinction.py`, and `subtraction.py` hold the
  image transforms and science calculations.
- `products.py` builds and writes output FITS products.
- `stages.py` registers the named pipeline stages.
- `runner.py` runs one galaxy or the configured sample.

The default stage order is:

```text
resolve_inputs
plan_outputs
skip_existing_outputs
load_images
load_errors
preprocess
resolve_bandpasses
calibrate_flux_density
apply_foreground_extinction
subtract_continuum
build_products
write_outputs
```

The default behavior is configured in `config/params.yaml`:

```yaml
contsub_pipeline:
  stages: null
  disabled_stages: []
```

Set `stages` to an explicit ordered list to run a custom workflow, or list
stage names under `disabled_stages` to remove them from the default sequence.
Unknown stage names raise a clear error before any galaxy is processed.

## Configuration

### Paths

Input and output roots live in `config/paths.yaml`:

```yaml
hst:
  image_root: "/lustre/opsw/work/abarnes/phangs/HST_WORK/scratch/HUBBLE_AND_CLUSTER_TECHNICAL_WORK/HST_image_products/HST_reduced_images"

contsub:
  output_root: "/lustre/opsw/work/abarnes/phangs/HST_WORK/data_contsub"
```

### File Discovery

The default HST input glob is in `config/files.yaml`:

```yaml
file_templates:
  hst_image_glob: "{hst.image_root}/{search_galaxy}/*{filter_digits}*/{search_galaxy}_*_{filter}_exp_drc_sci.fits"
  hst_error_glob: "{hst.image_root}/{search_galaxy}/*{filter_digits}*/{search_galaxy}_*_{filter}_err_drc_wht.fits"
```

For `ngc5068` and `f658n`, this becomes a search like:

```text
.../HST_reduced_images/ngc5068/*658*/ngc5068_*_f658n_exp_drc_sci.fits
```

### Galaxy Defaults And Overrides

The configured galaxy sample is listed in `config/galaxies.yaml`. Defaults are:

```yaml
defaults:
  broad_filters: ["f555w", "f814w"]
  narrow_filters: ["f657n", "f658n"]
  preferred_instruments: ["uvis", "acs"]
  hdu_index: 0
  overwrite: false
  write_continuum: true
  contsub_space: "linear"
  require_errors: true
  narrowband_width_header: "PHOTBW"
  narrowband_widths: {}
  nii_to_halpha: 0.25
  output_unit: "erg/s/cm2/arcsec2"
```

If file discovery finds more than one plausible file, add an override. Examples:

```yaml
overrides:
  ngc5068:
    f658n_file: "/full/path/to/ngc5068_uvis_f658n_exp_drc_sci.fits"
    f658n_error_file: "/full/path/to/ngc5068_uvis_f658n_err_drc_wht.fits"

  ngc2903:
    preferred_instruments: ["acs", "uvis"]

  ngc628c:
    search_galaxy: "ngc628c"
    sample_name: "ngc628"
    overwrite: true

  ngc4321:
    nii_to_halpha: 0.15
    narrowband_widths:
      f657n: 121.0
```

Use filter-specific keys for exact files:

```yaml
f555w_file: "/full/path/to/file.fits"
f814w_file: "/full/path/to/file.fits"
f657n_file: "/full/path/to/file.fits"
f658n_file: "/full/path/to/file.fits"
f555w_error_file: "/full/path/to/error_file.fits"
f814w_error_file: "/full/path/to/error_file.fits"
f657n_error_file: "/full/path/to/error_file.fits"
f658n_error_file: "/full/path/to/error_file.fits"
```

### Foreground Extinction

Foreground extinction is configured in `config/params.yaml`:

```yaml
foreground_extinction:
  enabled: true
  sample_table_path: "/path/to/phangs_sample_table_v1p6.fits"
  galaxy_column: "name"
  ebv_column: "mwext_sf11"
  r_v: 3.1
```

The code reads E(B-V), applies a CCM89 foreground correction at each filter's
resolved pivot wavelength, and propagates the same correction to the error
maps. If a galaxy uses a subfield name such as `ngc628c`, set `sample_name`
under that galaxy's override so the sample-table row can be found.

### Bandpasses

Bandpass metadata is configured in `config/params.yaml`:

```yaml
bandpass:
  filter_root: "/path/to/hst_filters"
  pivot_source: "filter"
  width_source: "filter_rectwidth"
  photflam_source: "header"
  table_file: "filter_table.fits"
  table_photflam_scale: 1.0e-19
  header_pivot_key: "PHOTPLAM"
  header_width_key: "PHOTBW"
  header_photflam_key: "PHOTFLAM"
  fallback_to_header: true
```

This mirrors the older `get_bandpassinfo(rootdir_bp)` workflow. The pipeline
reads every `*.dat` throughput curve in `filter_root` into a bandpass catalog
keyed by instrument/filter, e.g. `UVIS_F657N` or `ACS_F658N`. When `synphot` is
installed it stores the same quantities as the old helper: `equivwidth`,
`integrate`, `rmswidth`, `photbw`, `fwhm`, `rectwidth`, `pivot`, and
`unit_response`. A numerical fallback estimates the key values if `synphot` is
not available.

The source for each quantity is explicit:

- `pivot_source: "filter"` uses the filter-curve `pivot`.
- `width_source: "filter_rectwidth"` uses the filter-curve `rectwidth`.
- `photflam_source: "header"` uses `PHOTFLAM` from the image header.

You can instead use `filter_table` or `header` for pivot/width, and `header`,
`filter_table`, or `filter_unit_response` for PHOTFLAM. If external bandpass
information is unavailable and `fallback_to_header` is true, the code falls
back to the configured FITS header keywords.

Instrument/filter matching uses the image filename/header to choose keys like
`UVIS_F657N` or `ACS_F658N`, matching the convention used in the older modules.

### Fixed [NII] Correction

The fixed [NII] correction is controlled by `nii_to_halpha` in
`config/galaxies.yaml`. It is interpreted as total `[NII] / H-alpha`, and the
H-alpha product is:

```text
halpha = contsub / (1 + nii_to_halpha)
```

The default is `0.25`, following the fixed 25% correction adopted as a simple
first-pass setting. See the PHANGS-Hα processing discussion in
[Razza et al. 2026, arXiv:2604.25627](https://arxiv.org/abs/2604.25627).
Override `nii_to_halpha` globally or per galaxy if a different value is needed.

## Method

For each galaxy, the default stage sequence:

1. Resolves F555W, F814W, the first available narrowband among F657N/F658N, and
   matching inverse-variance error images.
2. Opens HDU 0 by default.
3. Converts exact zero science pixels to NaN and crops all images to remove
   NaN padding around the narrowband image.
4. Converts inverse-variance maps to 1-sigma errors with `sqrt(1 / weight)`.
5. Converts science and error images to flux density with
   `data * PHOTFLAM * 1e20`.
6. Applies the configured foreground extinction correction.
7. Resolves pivot wavelengths and the narrowband width from the configured HST
   filter table/curves, falling back to FITS headers if needed.
8. Computes a continuum estimate at the narrowband pivot wavelength. The
   default `contsub_space: "linear"` setting uses:

```text
continuum = weight_f555w * f555w + weight_f814w * f814w
```

Set `contsub_space: "log"` in `config/galaxies.yaml` defaults or a per-galaxy
override to use log-space interpolation instead:

```text
continuum = 10 ** (weight_f555w * log10(f555w) + weight_f814w * log10(f814w))
```

where the weights come from the resolved pivot wavelengths:

```text
weight_f555w = abs(pivot_f814w - pivot_narrow) / abs(pivot_f555w - pivot_f814w)
weight_f814w = abs(pivot_f555w - pivot_narrow) / abs(pivot_f555w - pivot_f814w)
```

9. Propagates the continuum and continuum-subtracted errors in quadrature.
10. Converts continuum-subtracted, continuum, and error maps from
   `erg/s/cm2/A/pixel` to integrated flux using the resolved narrowband width
   or a `narrowband_widths` override.
11. Divides by the FITS WCS pixel area so the final maps are surface brightness
    in `1e-20 erg/s/cm2/arcsec2` by default. Set `output_unit:
    "erg/s/cm2/pixel"` in `config/galaxies.yaml` to keep per-pixel fluxes.
12. Writes the raw continuum-subtracted flux products and the fixed-[NII]
    H-alpha products.
13. Writes `contsub_manifest.csv` summarizing inputs, outputs, weights, and failures.
14. Writes timestamped text and JSONL audit logs under `logs/`, including dry runs.

The code checks that all three image arrays have the same shape. If a target
fails because shapes differ, that galaxy needs a later reprojection step before
this plain subtraction can be applied.
