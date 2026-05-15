# hstha_contsubpipe_extended

A small, configuration-driven pipeline for first-pass HST narrowband continuum
subtraction of the PHANGS HST sample.

The current implementation keeps to a plain no-MUSE workflow: zero-to-NaN
preprocessing, narrowband padding removal, foreground extinction correction,
linear-space continuum interpolation between F555W and F814W, propagated error
maps, conversion to integrated flux, and a configurable fixed [NII] correction.
It does not yet do anchoring, PSF matching, reprojection, or log-space
subtraction.

## Directory Layout

- `config/paths.yaml` - root input and output directories.
- `config/files.yaml` - filename and glob templates.
- `config/galaxies.yaml` - galaxy list, defaults, and per-galaxy overrides.
- `src/hstha_contsubpipe_extended/contsub.py` - continuum-subtraction code and CLI.
- `src/hstha_contsubpipe_extended/tasks/run_contsub.py` - task-registry wrapper.

Outputs are written to:

```text
/lustre/opsw/work/abarnes/phangs/HST_WORK/data_contsub
```

The main products are:

```text
{galaxy}_{narrow_filter}_contsub_flux_linear.fits
{galaxy}_{narrow_filter}_contsub_flux_err_linear.fits
{galaxy}_{narrow_filter}_continuum_flux_linear.fits
{galaxy}_{narrow_filter}_continuum_flux_err_linear.fits
{galaxy}_{narrow_filter}_halpha_flux_nii_corr_linear.fits
{galaxy}_{narrow_filter}_halpha_flux_nii_corr_err_linear.fits
contsub_manifest.csv
```

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
`PHOTPLAM`, and propagates the same correction to the error maps. If a galaxy
uses a subfield name such as `ngc628c`, set `sample_name` under that galaxy's
override so the sample-table row can be found.

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

For each galaxy, the pipeline:

1. Resolves F555W, F814W, the first available narrowband among F657N/F658N, and
   matching inverse-variance error images.
2. Opens HDU 0 by default.
3. Converts exact zero science pixels to NaN and crops all images to remove
   NaN padding around the narrowband image.
4. Converts inverse-variance maps to 1-sigma errors with `sqrt(1 / weight)`.
5. Converts science and error images to flux density with
   `data * PHOTFLAM * 1e20`.
6. Applies the configured foreground extinction correction.
7. Computes a linear continuum estimate at the narrowband pivot wavelength:

```text
continuum = weight_f555w * f555w + weight_f814w * f814w
```

where the weights come from the `PHOTPLAM` values:

```text
weight_f555w = abs(PHOTPLAM_f814w - PHOTPLAM_narrow) / abs(PHOTPLAM_f555w - PHOTPLAM_f814w)
weight_f814w = abs(PHOTPLAM_f555w - PHOTPLAM_narrow) / abs(PHOTPLAM_f555w - PHOTPLAM_f814w)
```

8. Propagates the continuum and continuum-subtracted errors in quadrature.
9. Converts continuum-subtracted, continuum, and error maps from
   `erg/s/cm2/A/pixel` to integrated flux using the narrowband width from
   `PHOTBW` or `narrowband_widths`.
10. Divides by the FITS WCS pixel area so the final maps are surface brightness
    in `1e-20 erg/s/cm2/arcsec2` by default. Set `output_unit:
    "erg/s/cm2/pixel"` in `config/galaxies.yaml` to keep per-pixel fluxes.
11. Writes the raw continuum-subtracted flux products and the fixed-[NII]
    H-alpha products.
12. Writes `contsub_manifest.csv` summarizing inputs, outputs, weights, and failures.

The code checks that all three image arrays have the same shape. If a target
fails because shapes differ, that galaxy needs a later reprojection step before
this plain linear subtraction can be applied.
