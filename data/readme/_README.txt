# HST Narrowband Continuum Subtraction Pipeline
# Extended H-alpha Product Release

## Overview

This release contains HST narrowband continuum-subtracted H-alpha products
generated with the configuration-driven continuum-subtraction pipeline in
hstha_contsubpipe_extended.

The pipeline uses HST F555W and F814W broadband imaging to estimate the stellar
continuum at the HST narrowband wavelength, then subtracts that continuum from
the F657N or F658N image. Products are written as flux maps and matching
propagated uncertainty maps. The final H-alpha products include a fixed [NII]
correction and are reported in:

erg/s/cm2/arcsec2

This release does not apply MUSE anchoring. Instead, the products are separated
into two processing families using the suffix in the filename:

- *_log.fits: products for the galaxies defined in Chandar et al. (2025).
  Instead of rerunning the full MUSE-anchoring workflow, this release directly
  applies the published background and [NII] corrections from Table 4 to the
  HST data. This gives the same resulting correction as the Chandar et al.
  products while keeping the release workflow self-contained. In this product
  directory, some fields appear as separate HST product roots, e.g. ngc628c
  and ngc628e.
- *_linear.fits: linear-space continuum subtraction for the extended sample.
  These galaxies do not have full-coverage MUSE data, so no background
  correction is applied. The continuum is fit in linear space, which is more
  stable for this sample and gives better results, and a fixed 25% [NII]
  correction is applied.

## What Was Done

- Reprocessed the configured HST narrowband sample using the current staged
  hstha_contsubpipe_extended workflow.
- Produced continuum, continuum-subtracted, and [NII]-corrected H-alpha flux
  maps for each successfully processed target.
- Produced propagated uncertainty maps for each released product type where
  suitable input error maps were available.
- Applied zero-to-NaN preprocessing before continuum subtraction.
- Removed narrowband padding and masked products to the common valid coverage
  of the HST broadband and narrowband images.
- Applied foreground Milky Way extinction corrections using the PHANGS sample
  table, with E(B-V) taken from the mwext_sf11 column and R_V = 3.1.
- Used HST filter curves for pivot wavelengths and rectangular widths, and
  image-header PHOTFLAM values for calibration.
- Converted products to flux surface brightness in erg/s/cm2/arcsec2.
- Applied a fixed [NII] correction:

  H-alpha = continuum_subtracted_flux / (1 + [NII]/H-alpha)

## Product Filename Convention

Products have the form:

{galaxy}_{narrow_filter}_{product}_{subtraction_space}.fits

where:

- {galaxy}: target or HST field name, e.g. ngc3059, ngc628e, ngc7496
- {narrow_filter}: f657n or f658n
- {product}: one of the products listed below
- {subtraction_space}: linear or log

The released product types are:

- *_continuum_flux_linear.fits or *_continuum_flux_log.fits
  Estimated continuum flux in the narrowband filter.
- *_continuum_flux_err_linear.fits or *_continuum_flux_err_log.fits
  Propagated uncertainty on the estimated continuum.
- *_contsub_flux_linear.fits or *_contsub_flux_log.fits
  Narrowband minus estimated continuum flux before [NII] correction.
- *_contsub_flux_err_linear.fits or *_contsub_flux_err_log.fits
  Propagated uncertainty on the continuum-subtracted flux.
- *_halpha_flux_nii_corr_linear.fits or *_halpha_flux_nii_corr_log.fits
  H-alpha flux after fixed [NII] correction.
- *_halpha_flux_nii_corr_err_linear.fits or *_halpha_flux_nii_corr_err_log.fits
  Propagated uncertainty on the [NII]-corrected H-alpha flux.

Example files include:

- ngc3059_f657n_continuum_flux_linear.fits
- ngc3059_f657n_contsub_flux_linear.fits
- ngc3059_f657n_halpha_flux_nii_corr_linear.fits
- ngc628e_f658n_continuum_flux_log.fits
- ngc628e_f658n_contsub_flux_log.fits
- ngc628e_f658n_halpha_flux_nii_corr_log.fits
- ngc7496_f657n_halpha_flux_nii_corr_log.fits

Each science product normally has a matching *_err_* uncertainty product,
except for ngc5194 as described below.

## Configuration Summary

The default processing parameters for the extended sample are:

- Broadband filters: F555W and F814W
- Narrowband filters searched: F657N and F658N
- Preferred instruments: UVIS, then ACS
- Continuum subtraction space: linear
- Custom background corrections: none by default
- Coverage mask: enabled
- Coverage-mask closing size: 10 pixels
- Coverage-mask closing iterations: 5
- Error images required: yes
- Default [NII]/H-alpha: 0.25
- Output unit: erg/s/cm2/arcsec2

Foreground extinction correction is enabled for the release. Values are read
from:

phangs_sample_table_v1p6.fits

using:

- galaxy column: name
- E(B-V) column: mwext_sf11
- R_V: 3.1

For HST subfields with suffixes such as c, e, n, s, or w, the foreground
extinction lookup falls back to the base galaxy name.

## Chandar et al. (2025) Log-Space Products

The following product roots are the galaxies defined in Chandar et al. (2025)
and have filenames ending in *_log.fits:

- ic5332
- ngc1087
- ngc1300
- ngc1365
- ngc1385
- ngc1433
- ngc1512
- ngc1566
- ngc1672
- ngc2835
- ngc3351
- ngc3627
- ngc4254
- ngc4303
- ngc4321
- ngc4535
- ngc5068
- ngc628c
- ngc628e
- ngc7496

For these products, the published Chandar et al. (2025) background corrections
and [NII]/H-alpha values are applied directly to the HST data, rather than
performing the full MUSE-anchoring steps. The result is the same correction as
defined in the paper. Their per-galaxy [NII]/H-alpha values, narrowband filter
selections, preferred instruments, and background offsets are set individually
in config/galaxies.yaml.

## Linear-Space Extended Sample

The remaining configured product roots use the extended-sample processing:
linear continuum interpolation, no custom background correction because full
MUSE coverage is not available, foreground extinction from the PHANGS sample
table, and a fixed 25% [NII] correction, i.e. [NII]/H-alpha = 0.25.

These products have filenames ending in *_linear.fits.

Linear-space product roots in this release configuration are:

- ic1954
- ic5273
- ngc1068
- ngc1097
- ngc1317
- ngc1511
- ngc1546
- ngc1559
- ngc1637
- ngc1792
- ngc1808
- ngc1809
- ngc2090
- ngc2566
- ngc2775
- ngc2903
- ngc2997
- ngc2997e
- ngc2997w
- ngc3059
- ngc3137
- ngc3368
- ngc3507
- ngc3511
- ngc3521s
- ngc3596
- ngc3621
- ngc4298
- ngc4424
- ngc4496a
- ngc4536
- ngc4548
- ngc4569
- ngc4571
- ngc4579
- ngc4654
- ngc4689
- ngc4694
- ngc4731
- ngc4826
- ngc4941
- ngc4951
- ngc5042
- ngc5194
- ngc5248
- ngc5530
- ngc5643
- ngc628
- ngc6300
- ngc6744
- ngc685
- ngc7456
- ngc7793

### NGC 5194 data provenance and errors

The F555W and F814W broadband data and the F658N narrowband data for ngc5194
were not reduced as part of PHANGS. They were taken from the HST Legacy Archive
Hubble Heritage ACS/WFC mosaic of M51. Error maps are not provided for these
data by the HST archive, and uncertainties are therefore not propagated here
for ngc5194.

## Notes

- The suffix _log or _linear is part of the processing provenance and should be
  retained when comparing products.
- The *_halpha_flux_nii_corr_* files are the recommended H-alpha maps for
  science use when a fixed [NII] correction is appropriate.
- The *_contsub_flux_* files retain the continuum-subtracted narrowband flux
  before the fixed [NII] correction.
- The adopted fixed [NII]/H-alpha values are approximate correction factors.
  Users needing spatially varying [NII] corrections should apply their own
  correction to the continuum-subtracted products.
- Full run logs and manifest files are generated by the pipeline at run time
  and should be used as the audit trail for exact input paths, selected filters,
  calibration values, and per-target settings.

## Known Limitations and TODOs

The following improvements are not included in this release:

- PSF matching between the broadband and narrowband HST images.
- Background matching for the extended-sample linear products.
- Colour correction for compact stellar clusters.

Because these steps are not yet included, some minor artifacts may remain in the
images. Compact point sources can have sub-optimal continuum subtraction and may
appear as strong negative residuals. Galaxy centres can also show sub-optimal
continuum subtraction where extended bulge light remains visible. These effects
should be kept in mind when interpreting compact sources, bright galaxy centres,
or regions dominated by old stellar continuum.

Please contact Ashley Barnes with questions or concerns.
