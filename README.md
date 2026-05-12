# hstha_contsubpipe_extended

A small, configuration-driven pipeline for first-pass HST narrowband continuum
subtraction of the PHANGS HST sample.

The current implementation intentionally does the plain tested workflow only:
linear-space continuum interpolation between F555W and F814W, followed by
subtraction from F657N or F658N. It does not yet do anchoring, PSF matching,
reprojection, or log-space subtraction.

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
{galaxy}_{narrow_filter}_contsub_linear.fits
{galaxy}_{narrow_filter}_continuum_linear.fits
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
```

If file discovery finds more than one plausible file, add an override. Examples:

```yaml
overrides:
  ngc5068:
    f658n_file: "/full/path/to/ngc5068_uvis_f658n_exp_drc_sci.fits"

  ngc2903:
    preferred_instruments: ["acs", "uvis"]

  ngc628c:
    search_galaxy: "ngc628c"
    overwrite: true
```

Use filter-specific keys for exact files:

```yaml
f555w_file: "/full/path/to/file.fits"
f814w_file: "/full/path/to/file.fits"
f657n_file: "/full/path/to/file.fits"
f658n_file: "/full/path/to/file.fits"
```

## Method

For each galaxy, the pipeline:

1. Resolves F555W, F814W, and the first available narrowband among F657N/F658N.
2. Opens HDU 0 by default.
3. Converts each image to flux density with `data * PHOTFLAM * 1e20`.
4. Computes a linear continuum estimate at the narrowband pivot wavelength:

```text
continuum = weight_f555w * f555w + weight_f814w * f814w
```

where the weights come from the `PHOTPLAM` values:

```text
weight_f555w = abs(PHOTPLAM_f814w - PHOTPLAM_narrow) / abs(PHOTPLAM_f555w - PHOTPLAM_f814w)
weight_f814w = abs(PHOTPLAM_f555w - PHOTPLAM_narrow) / abs(PHOTPLAM_f555w - PHOTPLAM_f814w)
```

5. Writes `narrowband - continuum` as a float32 FITS image.
6. Writes `contsub_manifest.csv` summarizing inputs, outputs, weights, and failures.

The code checks that all three image arrays have the same shape. If a target
fails because shapes differ, that galaxy needs a later reprojection step before
this plain linear subtraction can be applied.
