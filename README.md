# Droplet Fusion

A Python analysis pipeline for epi-fluorescence microscopy TIFF time series of merging
(fusing) droplets. For each input movie it segments the droplet, fits an ellipse per frame,
measures the aspect ratio (AR) over time, fits the post-fusion AR relaxation, and estimates
the fused radius and inverse capillary velocity. It writes per-movie quality-control (QC)
images/CSVs and a dataset-level summary.

This guide assumes no prior Python setup. If you can open a terminal and copy-paste a few
commands, you can run it.

---

## 1. Install `uv` (one time)

This project uses [`uv`](https://docs.astral.sh/uv/), a single tool that installs the correct
Python version and all dependencies for you. You do **not** need to install Python yourself.

Open a terminal and run the line for your operating system:

- **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows** (PowerShell):

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Close and reopen the terminal, then confirm it worked:

```bash
uv --version
```

## 2. Get the code

Download this repository (or clone it with `git`) and open a terminal **inside the project
folder** — the folder that contains this `README.md` and a `pyproject.toml` file.

## 3. Install the project

From inside the project folder, run:

```bash
uv sync
```

This creates an isolated environment (`.venv/`), downloads the right Python, and installs
every dependency. It can take a few minutes the first time. You only need to do this once
(re-run it if the code is updated).

## 4. Run it on the bundled example data

The `data/` folder ships with several example `.tif` movies. Analyze them with:

```bash
uv run droplet-fusion \
    --data-dir data \
    --output-dir output \
    --seconds-per-frame 600 \
    --um-per-pixel 0.2125 \
    --overwrite
```

On Windows PowerShell, put it on one line (drop the `\` line-continuations).

While it runs, a live progress bar shows how many movies are done and the current stage of
the one being processed (loading → segmenting → measuring & fitting → rendering video), so you
can always see that the program is working:

```text
Found 9 movie(s) in data. Analyzing...
Analyzing movies:  67%|██████▋   | 6/9 [00:06<00:03,  1.1s/movie, example_07.tif: rendering video]
```

When it finishes you'll see a message listing what was written. Results land in the
`output/` folder (see below). Add `--no-make-videos` for a faster run that skips the
per-movie `overlay.mp4` files, and `--no-progress` to silence the progress bar (useful when
writing the terminal output to a log file).

## 5. Where the results go

Inside `--output-dir` you get **one folder per movie**, plus dataset-level summary files:

Per movie (e.g. `output/example01/`):

| File | What it is |
| --- | --- |
| `masks.tif` | The segmented droplet mask for every frame |
| `frame_measurements.csv` | Per-frame ellipse measurements (area, aspect ratio, etc.) |
| `segmentation_qc.csv` | Per-frame segmentation diagnostics |
| `AR_fit.png` | Plot of aspect ratio vs. time with the fitted relaxation curve |
| `movie_result.json` | Fitted parameters (relaxation time, fused radius, …) |
| `overlay.mp4` | Movie with the mask/ellipse drawn on top (unless `--no-make-videos`) |

Dataset-level (top of `--output-dir`):

| File | What it is |
| --- | --- |
| `summary_results.csv` | One row per movie with the key fitted numbers |
| `inverse_capillary_velocity_summary.png` | Summary plot across all movies |
| `run_config.json` | Every setting used for this run (for reproducibility) |
| `run_log.txt` | Log of what happened |

## 6. Run it on your own data

Put your `.tif` / `.tiff` movies in a folder and point `--data-dir` at it.

**What each movie must contain.** The pipeline expects each `.tif` / `.tiff` to be a
**single-channel** time series (a grayscale image stack, not multi-channel/RGB) cropped tightly
around a **single droplet pair**, where:

- the **first frame** shows the two droplets **already touching** (fusion has just begun), and
- the **last frame** shows them **fully merged into a single circular droplet**.

Partial droplets from neighboring pairs may appear in the background **only if** they touch the
image border and are spatially distinct (clearly separated) from the merging pair of interest;
in that case they will not interfere with the analysis. Anything that overlaps or connects to the
central pair will corrupt the measurements.

**Two settings must match your microscope**, or the physical numbers will be wrong:

- `--seconds-per-frame` — the time between frames, in seconds.
- `--um-per-pixel` — the pixel size, in microns.

```bash
uv run droplet-fusion \
    --data-dir /path/to/your/movies \
    --output-dir my_results \
    --seconds-per-frame 600 \
    --um-per-pixel 0.2125 \
    --overwrite
```

`--overwrite` lets the run replace results from a previous run in the same output folder.

## 7. See all options

```bash
uv run droplet-fusion --help
```

---

## Tuning and quality control (optional)

You can usually ignore this section at first. Come back to it if the automatic results look
off when you inspect the `AR_fit.png` and `overlay.mp4` files. Every option below can be added
to the `uv run droplet-fusion ...` command; `--help` prints this same list at any time.

Options written as `--flag / --no-flag` are on/off switches — pass the `--no-` form to turn the
default off (e.g. `--no-make-videos`).

**Segmentation.** By default (`--segmentation-mode temporal`) the pipeline evaluates several
candidate masks per frame and scores them by area, shape, border clearance, and continuity
with neighboring frames, writing a `segmentation_qc.csv` per movie. Add
`--outer-envelope-mode convex_hull` to fill mask concavities caused by internal dim bands;
it is opt-in because it can over-fill genuinely necked early-fusion shapes.

**How the fit window is chosen (and why it's robust).** The pipeline fits the post-fusion
aspect ratio to an exponential relaxation, `AR(t) = 1 + A·exp(−t/τ)`. Both ends of the fit
window are picked automatically so that the fitted relaxation time `τ` does not depend on how
each movie happened to be cropped in time:

- **Start — `--fit-start-mode decay_onset` (default).** A cropped movie does not necessarily
  begin exactly at the moment the droplets touch: it may open a few frames early, on a near-flat
  "pre-collapse plateau" where the aspect ratio barely changes, and the per-frame AR is noisy.
  `decay_onset` lightly smooths AR with a short (3-frame) moving average, finds the smoothed peak
  (maximum elongation), then walks *forward* to the first frame that has dropped about 5% of the
  way from that peak toward the post-fusion floor. That frame — the start of the *sustained*
  decay — becomes `t = 0`. This automatically detects the true onset of merger and is insensitive
  both to exactly where the movie was cropped at the start and to single-frame noise, unlike
  `max_AR`, which pins `t = 0` to one (noise-sensitive) peak frame.

- **End — `--trim-tail-to-floor` (default on) with `--tail-floor-fraction` (default `0.1`).**
  Once the droplets have merged, AR flattens near 1 (a circle) for the rest of the movie. Those
  near-circular frames carry almost no information about the relaxation but a lot of measurement
  scatter (worse at low resolution), and there can be many of them depending on how long the
  movie was cropped — so left in, they would dominate the least-squares fit and bias `τ`. The
  trim keeps frames only down to the first one that reaches within `--tail-floor-fraction` (10%
  by default) of the way from the top of the descent to its floor, then stops. The **last** frame
  used in the fit is therefore set by where the AR curve actually flattens, not by how many extra
  circular frames happen to be in the stack, making the end of the window robust to temporal
  cropping, noise, and resolution.

**Flagging and manual override.** Successful fits with `fit_r2 < 0.8` are flagged as a warning.
When a fit is flagged, inspect that movie's `AR_fit.png` and `overlay.mp4`, then supply reviewed
starting frames with `--fit-start-mode manual --manual-fit-starts starts.csv` (a CSV with
`filename,fit_start_frame` columns). The alternative `--fit-start-mode max_AR` starts at the peak
AR frame and, with `--fail-max-ar-at-last-frame` (default on), fails when that peak is the last
valid frame (no relaxation frames follow it).

### Input, units, and run control

| Option | Default | What it does |
| --- | --- | --- |
| `--data-dir PATH` | `data` | Folder of input `.tif` / `.tiff` movies. |
| `--output-dir PATH` | `output` | Folder where results are written. |
| `--seconds-per-frame FLOAT` | `1.0` | Time between frames, in seconds. **Set this to match your acquisition.** |
| `--um-per-pixel FLOAT` | `1.0` | Pixel size, in microns. **Set this to match your microscope.** |
| `--overwrite` | off | Allow replacing results from a previous run in the same output folder. |
| `--progress / --no-progress` | on | Show a live progress bar while movies are analyzed. Use `--no-progress` for quiet/log-file runs. |

### Segmentation

| Option | Default | What it does |
| --- | --- | --- |
| `--segmentation-mode {per_frame,temporal}` | `temporal` | `temporal` scores candidate masks using neighboring frames; `per_frame` scores each frame independently. |
| `--threshold-method {otsu,yen,li,triangle}` | `otsu` | Thresholding method for background-corrected frames. |
| `--snr-threshold FLOAT` | `3.0` | Minimum signal-to-noise ratio for accepting a frame. |
| `--min-object-area-px INT` | `100` | Smallest allowed droplet area, in pixels. |
| `--max-object-area-px INT` | none | Largest allowed droplet area, in pixels (no upper limit by default). |
| `--min-area-ratio-to-reference FLOAT` | `0.35` | Reject a frame whose mask is smaller than this fraction of the temporal reference. |
| `--max-centroid-jump-px FLOAT` | auto | Reject a frame whose droplet center jumps more than this many pixels (default: a fraction of the image diagonal). |
| `--mask-closing-radius-px INT` | `2` | Morphological closing radius applied to candidate masks. |
| `--boundary-threshold-percentile FLOAT` | `60.0` | Low corrected-intensity percentile used to grow boundary candidates. |
| `--outer-envelope-mode {none,convex_hull}` | `none` | `convex_hull` fills mask concavities from internal dim bands (can over-fill necked shapes). |
| `--background-poly-order INT` | `2` | Polynomial order for the background-illumination fit. |
| `--background-mask-dilation-px INT` | `10` | Foreground-mask dilation applied before background fitting. |
| `--write-segmentation-qc / --no-write-segmentation-qc` | on | Write per-movie `segmentation_qc.csv` diagnostics. |

### Fitting (AR relaxation)

| Option | Default | What it does |
| --- | --- | --- |
| `--fit-start-mode {max_AR,manual,first_valid,decay_onset}` | `decay_onset` | Which frame starts the fit window. `decay_onset`: top of the sustained decay, skipping pre-collapse plateaus (robust to cropping/noise). `max_AR`: the single peak-aspect-ratio frame. `first_valid`: the first measured frame. `manual`: read starts from `--manual-fit-starts`. |
| `--manual-fit-starts PATH` | none | CSV with `filename,fit_start_frame` columns. Required when `--fit-start-mode manual`. |
| `--trim-tail-to-floor / --no-trim-tail-to-floor` | on | Drop the near-circular AR tail so it doesn't dominate the fit residual. |
| `--tail-floor-fraction FLOAT` | `0.1` | Keep fit frames down to within this fraction of the descent floor before trimming the tail. |
| `--float-baseline / --no-float-baseline` | off | Fit `AR(t) = b + A·exp(−t/τ)` with a free asymptote `b ≥ 1` instead of pinning `b = 1`. |
| `--n-final-frames-for-r INT` | `5` | Number of final valid frames averaged to estimate the fused radius. |
| `--min-fit-r2-warning FLOAT` | `0.8` | Flag a successful fit as a `warning` when its R² is below this value. |
| `--fail-max-ar-at-last-frame / --no-fail-max-ar-at-last-frame` | on | Fail `max_AR` fits when the peak AR is the last valid frame (no relaxation frames follow it). |

### QC overlay videos

| Option | Default | What it does |
| --- | --- | --- |
| `--make-videos / --no-make-videos` | on | Write per-movie `overlay.mp4` QC videos. Turn off for faster runs. |
| `--video-fps FLOAT` | `10.0` | Frames per second for the overlay videos. |

## Running the tests (optional)

```bash
uv run pytest
```

## Troubleshooting

- **`uv: command not found`** — reopen your terminal after installing `uv` (step 1); the
  installer adds it to your `PATH` only in new terminals.
- **`No .tif/.tiff files found`** — check that `--data-dir` points at the folder that actually
  contains your movies.
- **Results replaced / "output already exists"** — add `--overwrite`, or choose a fresh
  `--output-dir`.

---

## References

This code was inspired by the following studies of DNA-droplet fusion and physical properties:

1. Sato, Y. & Takinoue, M. Sequence-dependent fusion dynamics and physical properties of DNA
   droplets. *Nanoscale Advances* **5**, 1919–1925 (2023).
   [doi:10.1039/D3NA00073G](https://doi.org/10.1039/D3NA00073G)
2. Chaderjian, A. S., Wilken, S. & Saleh, O. A. Diverse, distinct, and densely packed DNA
   nanostar droplets. *Proceedings of the National Academy of Sciences* **123**, e2523462123
   (2026). [doi:10.1073/pnas.2523462123](https://doi.org/10.1073/pnas.2523462123)
