---
title: 'FLIMKit: a Python toolkit for reconvolution fitting, lifetime-distribution fitting, and phasor analysis of FLIM data'
tags:
  - Python
  - FLIM
  - fluorescence lifetime
  - TCSPC
  - phasor
  - microscopy
authors:
  - name: Alexander Hunt
    orcid: 0009-0006-2990-649X
    corresponding: true
    affiliation: 1
  - name: Ava Russell
    orcid: 0009-0004-6340-1167
    affiliation: 1
  - name: Layla Mathieson
    orcid: 0000-0001-8258-7025
    affiliation: 1
  - name: Zhen Yuan Yeo
    orcid: 0000-0003-2766-6972
    affiliation: 2
  - name: Matthieu Veremen
    orcid: 0000-0003-3584-0387
    affiliation: 1
  - name: Ahsan Akram
    orcid: 0000-0003-4605-1682
    affiliation: 1
affiliations:
  - name: Centre for Inflammation Research, Institute for Regeneration and Repair, The University of Edinburgh, Edinburgh, United Kingdom
    index: 1
  - name: Department of Statistics and Data Science, National University of Singapore, Singapore
    index: 2
date: 21 October 2026
bibliography: paper.bib
---

# Summary

Fluorescence lifetime imaging microscopy (FLIM) measures how long a fluorophore stays excited before emitting a photon. The lifetime reports on the local environment of the molecule (pH, oxygen tension, ion binding, proximity to an acceptor in Förster resonance energy transfer) and not on concentration or excitation intensity [@datta2020]. The fluorophore can be a label or an endogenous species such as NAD(P)H and FAD, so FLIM applies label-free to tissue autofluorescence. That sensitivity underpins translational work: autofluorescence lifetime alone classifies human T cells by activation state with 97--99% accuracy [@walsh2021], discriminates lung cancer from adjacent tissue during fibre-based imaging [@fernandes2024], generates virtual haematoxylin-and-eosin histology from unstained tissue [@wang2024npj], and predicts EGFR mutation status in lung adenocarcinoma [@zang2026]. Doing this at scale needs analysis software that can be scripted, reproduced and used with any vendor's instrument.

FLIMKit (v0.14.1) is a Python toolkit for FLIM data from time-correlated single-photon counting (TCSPC) instruments [@shcheslavskiy2025]. PicoQuant `.ptu`, Becker & Hickl `.sdt`, Photonscore `.photons` and, experimentally, ISS files are read through one shared file abstraction. Three analysis routes are supported: reconvolution fitting of discrete multi-exponential models, fitting of continuous lifetime distributions, and calibrated phasor analysis [@digman2007; @ranjit2018]. Per-pixel fitting is accelerated on Apple Silicon, NVIDIA CUDA and AMD ROCm hardware. FLIMKit runs as a desktop application, a guided terminal interface, command-line scripts or a Python API, and exports OME-TIFF, GeoJSON for QuPath, and CSV.

# Statement of need

Laboratories running Leica FALCON/STELLARIS, Becker & Hickl SPC, Photonscore LINCam or ISS FastFLIM/Vista instruments mostly analyse their data with the vendor's software (LAS X, SPCImage, SymPhoTime). These packages are closed-source, graphical-only and tied to one file format, so batching, scripting and any analysis the vendor did not anticipate are hard, and a facility with more than one platform has no single reproducible pipeline. FLIMKit decodes every supported format into one `FLIMFile` abstraction, so the same fitting, phasor, stitching and export code runs whatever the instrument.

Open-source tools cover separate parts of the problem. FLIMfit [@warren2013] offers established reconvolution and global fitting but no phasor analysis, is built around PicoQuant and OME-TIFF data, and has had no release since 2017. PhasorPy [@phasorpy], the `napari-flim-phasor-plotter` and `napari-phasors` plugins [@napariflim; @napariphasors], FLIMPA [@flimpa2025] and GSLab [@gslab2025] implement the phasor method of Digman et al. [@digman2007], formalised as a fit-free protocol by Ranjit et al. [@ranjit2018]. None performs reconvolution fitting with full instrument-response-function (IRF) deconvolution, and GSLab needs MATLAB. FLIMKit uses PhasorPy for the phasor method itself and adds calibration, interactive cursors, spatial filtering and ROI-based batch export on top. We designed FLIMKit for laboratories and imaging facilities that need vendor-independent analysis across these routes, for users who open, fit and export through the interface without writing code and for those building automated pipelines.

PicoQuant, Becker & Hickl and Photonscore support has been validated against real instrument files. ISS support is built from the vendor's format specifications, has not been checked against real ISS acquisitions, and is experimental.

# State of the field

| Software | Access | Formats | Fitting + phasor | Scriptable |
|---|---|---|---|---|
| LAS X / SPCImage / SymPhoTime | Proprietary | Vendor-locked | Yes / partial | No |
| FLIMfit [@warren2013] | Open source | PicoQuant, OME-TIFF | Yes / no | Partial |
| PhasorPy [@phasorpy] | Open source | Many (readers) | No / yes | Yes |
| napari-flim-phasor-plotter / napari-phasors [@napariflim; @napariphasors] | Open source | Several | No / yes | Via napari |
| FLIMPA [@flimpa2025] | Open source | PTU, .sdt, .tif | No / yes (+ROI) | No |
| GSLab [@gslab2025] | Open source | Several | No / yes | Via MATLAB |
| FLIMKit | Open source | PTU, .sdt, Photonscore, ISS (exp.) | Yes / yes | Yes |

Table: Comparison of FLIM analysis software. FLIMKit's phasor analysis is built on PhasorPy.

To our knowledge FLIMKit is the only open-source package that supports several instrument vendors behind one abstraction, performs both reconvolution fitting and phasor analysis, and is scriptable with GPU acceleration. The fitting tools (FLIMfit; TRI2/FLIMLib, an ImageJ library) are largely unmaintained, and the phasor tools do not fit decays. The phasor packages are more mature in their own domain, so FLIMKit builds on PhasorPy. The contribution is integration: one file abstraction, both analysis families in one interface, and three validated vendors plus an experimental fourth.

# Software design

The central design decision is the `FLIMFile` abstraction. Where mature decoders exist the readers wrap `ptufile` and `sdtfile` (both by Gohlke) for PicoQuant and Becker & Hickl data and `photonsfile` [@photonsfile] for Photonscore, and reduce their output to a common `(Y, X, H)` decay-cube-and-metadata contract. No instrument-specific field passes further into the pipeline, which constrains new readers but gives one tested code path instead of one per format. Photonscore files record each photon with its own $(x, y)$ position and micro-time, so the image is formed by binning positions and the decay by histogramming micro-times. `photonsfile` was written for FLIMKit, released as a standalone package, and validated bit-exact against the vendor SDK.

Per-pixel fitting is the dominant computational cost: a $1024 \times 1024$ acquisition requires of the order of $10^6$ independent non-linear fits. No single GPU interface spans Apple Silicon, NVIDIA and AMD hardware, so a backend abstraction (MLX, PyTorch MPS/CUDA/ROCm, numba, or a CPU fallback) lets one `fit_per_pixel()` call run on whatever is available. Photon counts per pixel are low and Poisson distributed, so when the lifetimes are fixed the amplitudes are estimated by maximum likelihood under a Poisson model with a fitted constant offset. The global ('summed') fit stays on the CPU, where it is not a bottleneck, and reports standard errors for every parameter. For three-exponential fits the errors from the curvature of the objective can be several times too small, so we also resample the decay under Poisson noise and refit. On one synthetic three-exponential decay the relative error on $\tau_2$ was 5.8% from curvature and 24% from the bootstrap, against 51% observed over repeated noise realisations.

Two lifetime models cover two situations: a discrete multi-exponential model for a small fixed number of species, and a continuous Gaussian/Lorentzian $\alpha(\tau)$ distribution following Lakowicz [@lakowicz2006] for heterogeneous populations, such as autofluorescence, that a fixed-species model would misrepresent. A time-varying background correction follows Görlitz et al. [@gorlitz2017].

Multi-tile acquisitions are stitched before fitting. Tile positions come from the stage metadata and are refined by cross-correlation between neighbours, following Preibisch et al. [@preibisch2009], so fitting and phasor analysis see one continuous field of view. For batch ROI workflows, cell masks come from the summed intensity image using Cellpose-SAM [@cellposesam2025], and per-cell lifetime and phasor statistics are exported without a separate segmentation step (\autoref{fig:gui}).

![The FLIMKit desktop interface in Single FOV Fit mode. Five freehand regions of interest outline individual cells on the intensity image and the per-pixel lifetime map. Below are the summed decay with its reconvolution fit and IRF, and the weighted residuals. The Fit Summary lists the recovered tri-exponential parameters, both mean lifetimes, the fitted background and IRF shift, and reduced $\chi^2$. Per-ROI statistics are in the Regions table. Data are a PicoQuant `.ptu` acquisition ($1024 \times 1024$ pixels, $1.7 \times 10^{6}$ photons).\label{fig:gui}](paper_fig_gui.png)

Capability that a laboratory needs but a general toolkit should not carry goes in through a plugin registry. A plugin is a Python module that registers against one of seven hooks (analysis tools, ROI panel buttons, startup actions, file formats, format sniffers, phasor filters and per-pixel fit backends) and declares the registry API version it was written for; a mismatch is refused, and a plugin that fails on import cannot stop the application starting. FLIMKit's own tools use the same registry, and a fixture plugin is loaded by the test suite on every run, so a change that breaks the contract fails in continuous integration. QuPath and Fiji bridges, a time-resolved anisotropy package and an optional C and CUDA fit accelerator ship as separate packages on this basis.

The four entry points call the same fitting and phasor functions, so an interactive analysis can be reproduced in a script unchanged. The test suite has 74 test modules covering CPU and GPU numerical parity, ground-truth lifetime recovery and format decoding. Compiled desktop applications for macOS, Linux and Windows are built on each tagged release, and Docker images are maintained in a separate repository. Installation, entry points and deployment are in the documentation.

# Research impact statement

FLIMKit is the routine analysis route for FLIM data at the Centre for Inflammation Research, University of Edinburgh, where we have used it on PicoQuant and Leica FALCON/STELLARIS acquisitions, including work in preparation.

Format support was developed with the instrument vendors. Becker & Hickl support was built and checked against sample data and format documentation from the vendor, Photonscore support against a sample `.photons` file and SDK documentation from Photonscore, and ISS format specifications were provided by ISS after a discussion on the CONFOCALMICROSCOPY listserv. FLIMKit is listed among phasor-analysis tools in the PhasorPy documentation, and has been applied to independently acquired datasets at the CRUK Scotland Institute and the National University of Singapore. This use is early and exploratory and has not been validated as a deployment. Wider external adoption remains a priority.

# AI usage disclosure

Anthropic's Claude assisted with parts of the graphical user interface, the add-on system, application packaging, Docker and TrueNAS deployment scripting, and debugging, under the direction of the lead author (A.H.). Contributions by Z.Y.Y. were developed with assistance from OpenAI's GPT-5.6 Sol, run through Hermes Agent by Nous Research; Z.Y.Y. directed and reviewed that work and is responsible for it. All scientific design (fitting models, IRF handling, phasor methods, validation approach, architecture) is the authors' own. AI-assisted code was reviewed and tested by the authors before merging, through the test suite, the installation validation script and CPU/GPU parity tests. No AI tool generated or verified the scientific or statistical methods described here.

# Acknowledgements

We thank Christoph Gohlke for PhasorPy, `ptufile`, `lfdfiles` and `sdtfile`; Dr. Jens Balke and Enzo Marscheck (Becker & Hickl) for `.sdt` documentation and sample data; Photonscore GmbH for the LINCam SDK, a sample `.photons` file, and for open-sourcing the D7 format; and Jeff Liao (ISS) for ISS format specifications.

# References
