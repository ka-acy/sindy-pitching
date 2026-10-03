# PySINDy Biomechanical Modeling of Baseball Pitching and Batting

## Technical paper

The current write-up is `docs/technical_paper.md`; the built IEEE PDF is
`technical_paper_ieee.pdf` (`python scripts/build_paper_pdf.py`). Section 11 of the paper
gives the command for every number it reports.

The Stage B pitching pool is 140 clips, split 111 train / 29 held-out test
(`data/baseball/splits/pitching_local_*.csv`). The operative configuration must be passed
explicitly; the refresh script's default config does not reproduce the paper:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python src/preprocessing/refresh_baseball_pipeline.py \
  --config config/preprocessing.baseball.unaligned.handedness.yaml
```

Validate script syntax with:

```bash
./.venv/bin/python -m compileall src
```

## Step 0 — Scope Definition and Success Criteria

### 0.1 Project Scope Statement

This project develops a machine vision and sparse dynamical systems modeling pipeline to analyze human motion, with eventual application to baseball pitching and batting mechanics.

The modeling framework leverages Sparse Identification of Nonlinear Dynamical Systems (SINDy), implemented via PySINDy, to discover governing equations from motion-derived kinematic time-series data.

Because elite baseball biomechanics datasets are difficult to access and computational pipelines must be validated prior to domain deployment, this project is structured as a two-stage research workflow:

- **Stage A — Pipeline Development and Validation**  
  Conducted on the UTD Multimodal Human Action Dataset (UTD-MHAD), which contains general human motion performed by non-athlete subjects.

- **Stage B — Domain Translation to Baseball Biomechanics**  
  The validated modeling pipeline is transferred to baseball-specific motion data, including MLB broadcast footage or motion capture datasets, to study pitching and batting dynamics.

The UTD-MHAD dataset is therefore used exclusively for computational and methodological validation, not for drawing conclusions about elite baseball performance or sport-specific mechanics.

---

### 0.2 Research Objectives

#### Primary Objective
Develop a reproducible computational pipeline that converts motion-derived joint trajectories into sparse nonlinear dynamical equations and coefficient matrices.

#### Secondary Objectives
- Validate skeleton capture, trajectory conditioning, and coefficient-matrix generation
- Quantify how preprocessing choices affect derivative stability and sparse regression output
- Identify whether joint-derived state variables can produce stable sparse coefficient structures
- Evaluate robustness of equation discovery under noisy motion conditions
- Prepare the framework for elite baseball motion analysis

---

### 0.3 Two-Stage Experimental Design

#### Stage A — Pipeline Capture Validation

Purpose: Validate the capture, preprocessing, feature-generation, and sparse-modeling infrastructure.

Dataset characteristics:
- Non-athlete subjects
- General human actions
- Variable coordination levels
- Controlled capture environment

Key validation targets:
- Pose extraction stability
- Joint trajectory fidelity
- Derivative estimation reliability
- Sparse regression convergence
- Coefficient matrix generation
- Sensitivity of coefficients to preprocessing and feature choices

No sport-specific or athlete-performance conclusions are drawn at this stage.

---

#### Stage B — Baseball Domain Translation

Purpose: Apply the validated pipeline to sport-specific biomechanics.

Target motion classes:
- Pitching delivery mechanics
- Bat swing kinematics

Potential data sources:
- MLB broadcast video
- Motion capture datasets
- Biomechanics lab recordings
- Statcast or similar tracking systems (if accessible)

This stage produces baseball-specific dynamical insights.

---

### 0.4 Motion Classes Modeled

To maintain continuity between Stage A and Stage B, two coarse motion archetypes are defined:

#### Batting-like Motions
Proxy characteristics:
- Rapid torso rotation
- Bilateral arm extension
- Rotational kinetic chain transfer

Target mapping:
- Baseball swing mechanics
- Bat acceleration sequencing

---

#### Pitching-like Motions
Proxy characteristics:
- Overhead throwing or ballistic arm motion
- Trunk rotation and extension
- Sequential proximal-to-distal energy transfer

Target mapping:
- Pitch delivery kinematics
- Arm acceleration dynamics

---

### 0.5 Success Criteria

#### Stage A — Pipeline Validation Success Metrics

The project is considered computationally validated if:

1. Pose tracking produces continuous skeletal trajectories.
2. Joint paths reflect physically plausible motion.
3. Velocity and acceleration estimates are numerically stable after filtering.
4. PySINDy converges and produces coefficient matrices without pipeline failure.
5. Equation terms and coefficient matrices are trackable across preprocessing and feature settings.
6. Similar motion classes can be compared at the level of sparse structure, without claiming baseball equivalence.

---

#### Stage B — Domain Translation Success Metrics

Pipeline translation is successful if:

1. Baseball pose extraction produces usable skeleton trajectories.
2. Kinematic feature computation transfers without redesign.
3. SINDy converges on baseball motion data.
4. Discovered dynamics align with established biomechanics literature.

---

### 0.6 Assumptions

- Human motion can be approximated as a nonlinear dynamical system.
- Skeletal kinematics sufficiently capture biomechanical state variables.
- Sparse regression can recover governing motion equations.
- Generic captured motions are sufficient to validate the mechanics of the computational pipeline before baseball-domain translation.

---

### 0.7 Known Limitations

- Proxy dataset subjects are non-athletes.
- Motions are not baseball-specific.
- Coordination differs from elite performance.
- Absolute biomechanical conclusions cannot be drawn from Stage A.
- Stage A equations and coefficients are infrastructure outputs, not baseball findings.

These limitations are mitigated by restricting Stage A conclusions to computational validation only.

---

### 0.8 Deliverables of Step 0

Completion of Step 0 produces:

- Defined experimental scope
- Two-stage research structure
- Motion class taxonomy
- Validation metrics
- Translation success criteria
- Documented assumptions and limitations

This scope definition governs all downstream implementation decisions.


---

## Step 1 — Dataset Acquisition and Organization

### 1.1 Dataset Used in Stage A (Proxy Validation)

**Dataset:** UTD Multimodal Human Action Dataset (UTD-MHAD)  
Source: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

**Project Role**

UTD-MHAD is used strictly for computational and pipeline validation.  
It contains general human motion performed by non-athlete subjects and does not represent elite baseball biomechanics.

This dataset enables development and testing of:

- Pose and skeleton extraction
- Joint trajectory tracking
- Kinematic feature computation
- Derivative estimation
- Sparse dynamical system identification via PySINDy

No sport-performance conclusions are drawn from this dataset. The primary Stage A outputs are stable processed trajectories, state matrices, derivatives, and sparse coefficient matrices.

---

### 1.2 Download and Local Storage

1. Download the dataset from the official source.
2. Extract all files.
3. Place them into the repository using the following path:
data/utd_mhad/raw/


Raw files must remain unchanged.  
All preprocessing outputs must be written to separate directories.

---

### 1.3 Repository Data Layout

The project should use the following standardized structure:

data/
utd_mhad/
raw/ # Original dataset (read-only)
processed/ # Cleaned trajectories + derived features
splits/ # Train/test and proxy subset manifests
metadata/ # Dataset index files

docs/papers/ # Research papers and notes

results/plots/equations/

src/pose/preprocessing/kinematics/sindy/eval/


This structure ensures reproducibility and portability to MLB datasets later.

---

### 1.4 Dataset Index Creation

A machine-readable dataset index must be created to track all trials and modalities.

Create the file:
'data/utd_mhad/metadata/index.csv'


Recommended columns:

- `subject_id`
- `action_id`
- `action_name`
- `trial_id`
- `modality` (rgb, depth, skeleton, imu, etc.)
- `filepath`
- `fps`
- `num_frames`

Purpose:

- Enables automated loading
- Prevents manual file searching
- Supports batch modeling workflows
- Standardizes data ingestion

If filenames encode metadata, still generate `index.csv` for script compatibility.

---

### 1.5 Proxy Motion Subset Definition

Because the dataset is not baseball-specific, a subset must be defined to approximate:

- Batting-like motions
- Pitching-like motions

Create the file:
'data/utd_mhad/splits/proxy_subset.csv'


Required columns:

- `subset_label` → batting_like or pitching_like
- `subject_id`
- `action_id`
- `trial_id`

Purpose:

- Documents motion selection logic
- Enables reproducible experiments
- Supports methodological defensibility in the paper

---

### 1.6 Smoke Test Subset (Fast Iteration Set)

Before processing the full dataset, define a small debugging subset.

Create:
data/utd_mhad/splits/smoke_test.csv


Use the same columns as `proxy_subset.csv`.

Recommended size:

- 10–30 trials total
- Balanced across batting-like and pitching-like motions

Purpose:

- Rapid pipeline testing
- Faster debugging
- Reduced compute overhead during development

---

### 1.7 Step 1 Success Criteria

Step 1 is complete when all of the following are true:

- Dataset exists in `data/utd_mhad/raw/`
- `metadata/index.csv` has been created
- `splits/proxy_subset.csv` has been defined
- `splits/smoke_test.csv` has been defined
- Repository folder structure matches specification

No modeling or preprocessing occurs in this step.

This stage strictly establishes data accessibility, organization, and experimental reproducibility.

---

---

## Step 2 — Proxy Action Selection Methodology

### 2.1 Purpose of Proxy Action Selection

The UTD-MHAD dataset does not contain baseball-specific motions.  
Therefore, a proxy action selection process is required to identify movements that biomechanically approximate pitching and batting dynamics.

This step defines a defensible mapping between general human actions and baseball motion archetypes to support dynamical modeling via PySINDy.

No claim is made that proxy motions replicate elite baseball mechanics.  
They are used solely to validate the modeling pipeline on rotational and ballistic human movement patterns.

---

### 2.2 Motion Archetype Framework

Two biomechanical archetypes are defined to maintain continuity between proxy and target datasets.

#### Batting-Like Motions

Characterized by:

- Rapid trunk rotation
- Bilateral arm extension
- Rotational kinetic chain transfer
- Distal segment acceleration
- Coordinated hip–torso–shoulder sequencing

These dynamics approximate baseball swing mechanics.

---

#### Pitching-Like Motions

Characterized by:

- Overhead or ballistic arm acceleration
- Sequential proximal-to-distal energy transfer
- Trunk rotation and extension
- Shoulder internal rotation acceleration
- Rapid upper limb angular velocity

These dynamics approximate pitch delivery mechanics.

---

### 2.3 Inclusion Criteria

An action is selected as a proxy motion if it demonstrates most of the following:

- Observable trunk rotational displacement
- Upper limb angular acceleration
- Multi-segment coordination
- Ballistic or explosive movement
- Kinetic chain sequencing
- Non-periodic motion patterns

Actions meeting these criteria provide sufficient dynamical richness for sparse equation discovery.

---

### 2.4 Exclusion Criteria

Actions are excluded if they primarily involve:

- Walking or locomotion
- Static gestures
- Sitting or standing motions
- Lower-body dominant movements
- Small-amplitude arm gestures
- Repetitive low-velocity activities

These motions lack biomechanical characteristics relevant to pitching or batting.

---

### 2.5 Action Labeling Process

Each selected trial is labeled as one of the following:

- `batting_like`
- `pitching_like`

Labeling is documented in:
data/utd_mhad/splits/proxy_subset.csv


This file enables reproducible experiment selection and transparent methodology reporting.

---

### 2.6 Methodological Justification

Proxy action selection is grounded in biomechanical principles established in the cited literature, including:

- Rotational kinetic chain sequencing
- Angular velocity transfer
- Trunk–shoulder coupling
- Distal segment acceleration

By modeling these shared dynamical structures, the pipeline can be validated prior to application on elite baseball datasets.

---

### 2.7 Step 2 Success Criteria

Step 2 is complete when:

- Motion archetypes are formally defined
- Inclusion and exclusion criteria are documented
- Proxy actions are identified within the dataset
- Trials are labeled as batting-like or pitching-like
- `proxy_subset.csv` is created

This step ensures biomechanical relevance of the proxy dataset prior to kinematic modeling.

---

---

## Step 3 — Pose Representation and Skeleton Standardization

### 3.1 Purpose

This step defines the biomechanical pose representation used throughout the modeling pipeline.

Raw motion data must be converted into standardized skeletal joint trajectories before kinematic features and dynamical models can be computed.

PySINDy operates on time-series state variables, not video data.  
Therefore, consistent pose representation is required to enable equation discovery.

---

### 3.2 Pose Data Sources

#### Dataset-Provided Skeletons (Primary Source)

The UTD-MHAD dataset includes Kinect-based skeletal joint tracking.

Advantages:

- Precomputed joint trajectories
- Synchronized multimodal capture
- Reduced noise compared to vision-only estimation
- Faster pipeline development

These skeletons serve as the primary pose representation for Stage A validation.

---

#### Vision-Based Pose Estimation (Future Translation)

For MLB or broadcast datasets where skeletons are not provided, pose estimation must be performed using computer vision models.

Potential tools include:

- MediaPipe Pose
- OpenPose
- DeepLabCut
- YOLO Pose variants

The pipeline is designed so that either dataset-provided or vision-derived skeletons can be used interchangeably.

---

### 3.3 Skeleton Schema Definition

A standardized joint schema must be defined to ensure consistency across datasets.

Recommended joint set:

- Head
- Neck
- Torso / Spine
- Left Shoulder
- Right Shoulder
- Left Elbow
- Right Elbow
- Left Wrist
- Right Wrist
- Left Hip
- Right Hip
- Left Knee
- Right Knee
- Left Ankle
- Right Ankle

Each joint must include 3D coordinates:
(x, y, z)


All joints must be consistently named and indexed across trials.

---

### 3.4 Coordinate Frame Normalization

Raw skeletal coordinates are camera-relative and must be normalized to represent biomechanical motion.

#### Translation Normalization

- Skeletons are centered at the pelvis or torso origin.
- Global subject movement is removed.
- Only relative joint motion is retained.

Purpose:

- Isolates biomechanical mechanics
- Improves modeling consistency

---

#### Orientation Normalization (Optional but Recommended)

Skeletons may be rotated into a canonical facing direction.

Purpose:

- Aligns subjects across trials
- Reduces viewpoint-induced variance
- Improves transferability to MLB datasets

---

### 3.5 Temporal Standardization

All pose sequences must have consistent temporal structure.

Requirements:

- Uniform timestep (dt)
- Known frame rate
- Continuous joint tracking

If necessary:

- Resample sequences
- Interpolate missing frames
- Trim inconsistent segments

This ensures stable derivative estimation in later steps.

---

### 3.6 Output of Step 3

Completion of this step produces:

- Standardized skeletal joint trajectories
- Normalized coordinate frames
- Dataset-agnostic pose representations
- Time-aligned motion sequences

These outputs serve as inputs for preprocessing and kinematic feature extraction.

---

### 3.7 Step 3 Success Criteria

Step 3 is complete when:

- Skeleton joint schema is defined
- Joint coordinate format is standardized
- Translation normalization is implemented
- Orientation normalization strategy is defined
- Temporal consistency is ensured

No kinematic features or dynamical models are computed in this step.

---

---

## Step 4 — Preprocessing and Time-Series Conditioning

### 4.1 Purpose

This step cleans and conditions skeletal joint trajectories to ensure they are numerically stable for:

- kinematic feature computation (angles, velocities, accelerations)
- derivative estimation
- sparse dynamical modeling via PySINDy

SINDy is highly sensitive to noise in time-series derivatives.  
Therefore, preprocessing is treated as a required modeling component, not an optional data cleanup step.

---

### 4.2 Inputs and Outputs

**Inputs (from Step 3):**
- Standardized skeletal joint trajectories (x, y, z per joint)
- Time base information (fps or dt)

**Outputs (for Step 5 and Step 6):**
- Cleaned, time-aligned joint trajectories written to: data/utd_mhad/processed/

Raw data in `data/utd_mhad/raw/` remains unchanged.

---

### 4.3 Time Base Standardization

All sequences must share a consistent time representation.

Requirements:

- Known frame rate (fps) or timestep (dt)
- Uniform sampling within each clip

If sampling is inconsistent:

- resample joint trajectories to a fixed dt
- store dt explicitly alongside each processed sample

---

### 4.4 Missing Data Handling

Skeletal tracking may include missing joints or dropped frames.

Rules:

- Interpolate short gaps (preferred)
- If gaps exceed a defined threshold, drop the clip or trim the affected segment
- Log all dropped or repaired segments for reproducibility

---

### 4.5 Noise Filtering and Smoothing

Joint trajectories must be smoothed before computing derivatives.

Recommended strategy:

- Apply smoothing per joint coordinate time-series
- Use a filter appropriate for derivative stability (e.g., Savitzky–Golay)
- Optionally apply a low-pass filter to suppress high-frequency tracking jitter

Filtering must be applied consistently across all clips used in modeling.

---

### 4.6 Clip Segmentation (Motion-Only Windows)

Clips may contain dead time before or after the main motion.

Preprocessing should:

- trim leading/trailing frames with minimal movement
- ensure the retained segment contains the primary ballistic/rotational action
- keep segmentation rules consistent across trials

This improves model consistency and reduces spurious dynamics.

---

### 4.7 Optional Normalizations

Depending on dataset variability, optional normalizations may be applied:

- **Translation normalization:** already performed in Step 3 (pelvis/torso-centered)
- **Scale normalization:** normalize by a body-size proxy (e.g., shoulder width) if inter-subject scale variation is large
- **Orientation normalization:** if enabled in Step 3, ensure consistent facing direction

Any enabled normalization must be documented and applied uniformly.

---

### 4.8 Reproducibility Requirements

Preprocessing must be deterministic and reproducible:

- all parameters (window sizes, cutoffs, thresholds) must be logged
- processed outputs should be versioned by configuration
- scripts must support re-running preprocessing from raw inputs

Recommended artifacts:

- preprocessing config file (JSON/YAML)
- run logs describing filtering and trimming decisions

---

### 4.9 Step 4 Success Criteria

Step 4 is complete when:

- all selected trials can be loaded from `index.csv`
- processed joint trajectories are written to `data/utd_mhad/processed/`
- time base is consistent and explicit (fps or dt)
- missing data rules are implemented and logged
- smoothing/filtering is applied consistently
- clips are trimmed to motion-relevant windows

No kinematic features or SINDy models are computed in this step.

---

---

## Step 5 — Kinematic Feature Extraction

### 5.1 Purpose

This step converts cleaned skeletal joint trajectories into biomechanical state variables suitable for sparse dynamical modeling.

PySINDy models dynamical systems of the form:

dX/dt = f(X)

Therefore, a meaningful state vector X(t) must be defined before equation discovery can occur.

---

### 5.2 Design Principles

Feature extraction must satisfy the following constraints:

- Dataset-agnostic (usable for both UTD-MHAD and MLB data)
- Biomechanically meaningful
- Robust to camera viewpoint
- Computable from standardized skeletons
- Minimal dimensionality to avoid overfitting

State variables should primarily use relative geometry rather than global camera coordinates.

---

### 5.3 Feature Categories

#### 1. Rotational Variables

Capture kinetic chain dynamics.

Examples:

- Trunk rotation angle
- Trunk angular velocity
- Shoulder rotation angle
- Shoulder angular velocity

These variables reflect proximal segment dynamics critical to pitching and batting.

---

#### 2. Upper Limb Velocity Variables

Capture distal acceleration and transfer.

Examples:

- Elbow angular velocity
- Wrist linear velocity
- Hand linear velocity

These approximate bat speed (batting-like motions) or release acceleration (pitching-like motions).

---

#### 3. Inter-Segment Coupling Variables

Capture coordination and sequencing.

Examples:

- Hip–shoulder separation angle
- Shoulder–elbow lag
- Torso–arm relative orientation

These allow the model to identify multi-segment coupling terms.

---

#### 4. Optional Energy Proxies

Nonlinear features may include:

- Squared angular velocities
- Linear velocity magnitude
- Segment kinetic energy proxy

These can improve discovery of nonlinear dynamical structure.

---

### 5.4 Recommended Initial State Vector Size

To maintain model stability, begin with a limited state dimension.

Recommended range:

6–12 state variables

Avoid high-dimensional feature sets during initial experiments.

Additional variables may be added after baseline validation.

---

### 5.5 Output Artifacts

Feature extraction produces:

- State matrix X (shape: time_steps × num_features)
- Time vector t or constant timestep dt
- Feature definition documentation

Saved under:
data/utd_mhad/processed/


Feature definitions should be logged in:
docs/feature_definitions.md


---

### 5.6 Step 5 Success Criteria

Step 5 is complete when:

- A biomechanically justified state vector is defined
- Feature extraction functions are implemented
- X(t) is generated for all selected trials
- Feature dimensionality is controlled and documented
- Output files are reproducible

No dynamical modeling occurs in this step.

---

---

## Step 6 — Derivative Estimation Strategy

### 6.1 Purpose

This step computes time derivatives of the state variables:

dX/dt

These derivatives are required for sparse dynamical system identification using PySINDy.

Accurate derivative estimation is critical.  
Noisy or unstable derivatives will result in incorrect or non-interpretable governing equations.

---

### 6.2 Inputs and Outputs

**Inputs (from Step 5):**

- State matrix X(t)
- Time vector t or timestep dt

**Outputs:**

- Derivative matrix dX/dt

Saved alongside feature data in:
data/utd_mhad/processed/


---

### 6.3 Challenges in Derivative Estimation

- Human motion data contains noise and tracking errors
- Numerical differentiation amplifies high-frequency noise
- Poor derivatives lead to unstable SINDy models

Therefore, smoothing and differentiation must be carefully combined.

---

### 6.4 Recommended Approach: Smoothed Finite Differences

The default method combines smoothing with numerical differentiation.

Procedure:

1. Apply smoothing to each feature time-series
2. Compute derivatives using finite differences

Recommended smoothing method:

- Savitzky–Golay filter

Advantages:

- Preserves motion shape
- Reduces noise without excessive distortion
- Supports stable derivative estimation

---

### 6.5 Alternative Methods (Optional)

PySINDy provides built-in differentiation options:

- FiniteDifference
- SmoothedFiniteDifference
- TotalVariationRegularizedDerivative

These methods may be explored after baseline validation.

---

### 6.6 Time Step Consistency

Derivative computation requires a consistent timestep.

Requirements:

- dt must be known and constant within each sequence
- dt must be correctly passed to all derivative functions

If sequences were resampled in Step 4, dt must reflect the resampled rate.

---

### 6.7 Consistency Requirements

All data must use:

- the same smoothing parameters
- the same differentiation method
- the same dt handling strategy

This ensures comparability across trials and reproducibility of results.

---

### 6.8 Validation of Derivatives

Before modeling, derivatives must be validated:

- velocity curves should be smooth and continuous
- acceleration curves should be physically plausible
- no large spikes or discontinuities should be present

If instability is observed:

- revisit preprocessing (Step 4)
- adjust smoothing parameters

---

### 6.9 Step 6 Success Criteria

Step 6 is complete when:

- dX/dt is computed for all selected trials
- derivative estimation is stable and consistent
- dt is correctly defined and used
- derivative quality has been visually or numerically validated

These outputs are required for PySINDy modeling in the next step.

---

---

## Step 7 — Sparse Dynamical Modeling with PySINDy

### 7.1 Purpose

This step applies Sparse Identification of Nonlinear Dynamical Systems (SINDy) to discover governing equations of motion from joint-derived state variables.

The objective is to model:

dX/dt = f(X)

where X represents feature vectors derived from captured joint motion.

---

### 7.2 Inputs and Outputs

**Inputs:**

- State matrix X(t)
- Derivative matrix dX/dt
- Time step dt

**Outputs:**

- Sparse dynamical models
- Discovered governing equations
- Coefficient matrices
- Model configurations

Saved under:
results/equations/


---

### 7.3 Model Structure

SINDy constructs a library of candidate functions from the state variables.

Initial recommended library:

- Polynomial terms (degree 2 or 3)

Example candidate terms:

- x₁, x₂
- x₁², x₂²
- x₁x₂

Optional extensions (later stages):

- Trigonometric functions (sin, cos)
- Higher-order nonlinear terms

---

### 7.4 Sparse Regression

SINDy applies sparse regression to select a minimal set of terms that explain the dynamics.

Key parameter:

- sparsity threshold

Guidelines:

- Lower threshold → more terms (risk of overfitting)
- Higher threshold → fewer terms (risk of underfitting)

Parameter tuning is required to balance interpretability and accuracy.

---

### 7.5 Separate Models by Motion Class

Models should be trained independently for:

- Batting-like motions
- Pitching-like motions

This enables comparison of dynamical structure between motion types.

---

### 7.6 Training Procedure

For each motion class:

1. Load X(t) and dX/dt
2. Define feature library
3. Initialize SINDy model
4. Fit model to data
5. Print and store discovered equations

Example (conceptual):

```python
model.fit(X, t=dt)
model.print()

---

### 7.7 Model Validation (Preliminary)

Basic validation should include:

- Checking equation sparsity
- Ensuring coefficients are stable across trials
- Verifying that dominant terms are physically plausible


### 7.8 Reproducibility

All modeling runs must log:
- library configuration
- sparsity threshold
- dataset subset used
- random seeds (if applicable)

This ensures experiments can be reproduced and compared.

### 7.9 Step 7 Success Criteria

Step 7 is complete when:
- SINDy models are successfully trained
- Sparse equations are produced
- Models are stored and documented

Separate models exist for batting-like and pitching-like motions

In Stage A, this step primarily produces the first reproducible sparse coefficient matrices and equations that can be used to validate the capture-to-model pipeline.

---

## Step 8 — Model Evaluation and Validation

### 8.1 Purpose

This step evaluates the validity of the discovered dynamical models.

For Stage A, the goal is to determine whether the equations produced by PySINDy:

- are computationally stable
- can be compared across trials and preprocessing settings
- yield usable coefficient structures for downstream analysis

---

### 8.2 Inputs and Outputs

**Inputs:**

- Trained SINDy models
- State variables X(t)
- Derivative data dX/dt

**Outputs:**

- Reconstruction plots
- Prediction results
- Error metrics
- Equation comparisons

Saved under:
results/plots/
results/evaluation/


---

### 8.3 Reconstruction Testing

Reconstruction evaluates whether the discovered model is numerically usable for short-horizon replay tests.

Procedure:

1. Use initial state X(0)
2. Simulate forward using the SINDy model
3. Compare simulated trajectory with actual data

Evaluation criteria:

- Stable short-horizon simulation behavior
- Lack of immediate blow-up
- Useful relative comparison across configurations

---

### 8.4 Generalization Testing

Generalization evaluates whether the sparse structure and derivative fit remain usable on different held-out trials.

Procedure:

- Train models on a subset of trials
- Test on different trials

Evaluation criteria:

- Reasonable derivative fit on held-out trials
- Gradual divergence rather than immediate instability in short rollouts
- Consistent coefficient behavior across subjects

---

### 8.5 Structural Consistency Analysis

This analysis evaluates whether discovered equations are stable across trials.

Checks include:

- Frequency of selected terms
- Consistency of dominant variables
- Similarity of coefficient structure

For Stage A, consistent equations indicate pipeline stability and recoverable sparse structure, not sport-specific truth.

---

### 8.6 Biomechanical Plausibility

Model outputs may be compared to broad biomechanical principles, but this is secondary in Stage A because the dataset is not athlete-specific. Relevant checks include:

- Kinetic chain sequencing (proximal-to-distal transfer)
- Trunk rotation influence on upper limb motion
- Angular velocity relationships between segments

The goal is not perfect alignment, but reasonable consistency with established theory.

---

### 8.7 Evaluation Metrics

Recommended metrics:

- Mean Squared Error (MSE)
- Trajectory overlap (visual comparison)
- Simulation stability
- Term frequency analysis

Visual inspection is an important component of validation, but Stage A interpretation should remain computational rather than baseball-specific.

---

### 8.8 Step 8 Success Criteria

Step 8 is complete when:

- short-horizon reconstruction tests are computationally informative
- models demonstrate some generalization capability at the derivative or sparse-structure level
- equation structures and coefficient matrices are comparable across trials
- the pipeline yields stable validation artifacts for later baseball translation

This step confirms that the modeling pipeline is functioning correctly as an infrastructure and capture-validation system.

---

---

## Step 9 — Equation Interpretation and Biomechanical Analysis

### 9.1 Purpose

This step interprets the discovered dynamical equations in biomechanical terms.

In Stage A, the goal is narrower: translate mathematical models into interpretable summaries of what the pipeline is capturing and whether the resulting sparse structures are stable enough to justify Stage B transfer.

---

### 9.2 Inputs and Outputs

**Inputs:**

- Sparse dynamical equations (Step 7)
- Evaluation results (Step 8)
- Feature definitions (Step 5)

**Outputs:**

- Interpreted equations
- Biomechanical insights
- Comparative analysis between motion types
- Written summaries for research reporting

Saved under:
results/analysis/
docs/


---

### 9.3 Dominant Variable Analysis

Identify which state variables appear most frequently and with the largest coefficients.

These variables represent dominant drivers of motion.

Examples:

- Trunk rotation variables
- Shoulder angular velocity
- Hand velocity

Dominant variables indicate key biomechanical contributors.

---

### 9.4 Coupling and Interaction Terms

Examine interaction terms in the equations.

Examples:

- x₁x₂ (cross-variable coupling)
- nonlinear dependencies between segments

These terms represent coordination between body segments and reflect kinetic chain behavior.

---

### 9.5 Nonlinear Dynamics

Identify nonlinear terms such as:

- squared variables (x²)
- cross-product terms (x₁x₂)

These indicate:

- nonlinear energy transfer
- complex coordination patterns
- higher-order biomechanical interactions

---

### 9.6 Comparison Between Motion Classes

Compare models trained on:

- Batting-like motions
- Pitching-like motions

Key comparisons:

- dominant variables
- strength of rotational terms
- presence of distal acceleration components

This analysis highlights differences in dynamical structure between motion types.

---

### 9.7 Biomechanical Alignment

Interpret results in the context of established biomechanics literature.

Relevant concepts include:

- kinetic chain sequencing
- proximal-to-distal energy transfer
- trunk rotation influence on upper limb motion
- angular velocity relationships

The objective is to identify qualitative alignment with known biomechanical principles.

---

### 9.8 Limitations

Interpretations must acknowledge:

- use of non-athlete proxy data
- absence of sport-specific motion
- potential noise in pose estimation

Results represent structural dynamical and computational insights, not elite performance analysis.

---

### 9.9 Step 9 Success Criteria

Step 9 is complete when:

- equations are translated into biomechanical terms
- dominant variables and interactions are identified
- motion classes are compared
- interpretations are aligned with literature where appropriate
- limitations are clearly stated

This step connects the computational pipeline to meaningful scientific conclusions.

---

---

## Step 10 — Translation to Baseball (MLB) Motion Data

### 10.1 Purpose

This step applies the validated biomechanical modeling pipeline to baseball-specific motion data.

The objective is to demonstrate that the pipeline developed using proxy motion data can be transferred to real-world sports contexts.

---

### 10.2 Pipeline Reuse

The pipeline developed in previous steps is reused without modification.

Reused components include:

- skeleton standardization
- preprocessing and filtering
- kinematic feature extraction
- derivative estimation
- PySINDy modeling
- evaluation and interpretation

Only the pose input layer is replaced.

---

### 10.3 Pose Extraction from Baseball Video

Unlike UTD-MHAD, baseball datasets typically do not include precomputed skeletons.

Pose estimation must be performed using computer vision models.

Potential tools:

- MediaPipe Pose
- OpenPose
- DeepLabCut
- YOLO Pose variants

Output:

- 2D or 3D skeletal joint trajectories compatible with Step 3 schema

---

### 10.4 Data Challenges

Baseball data introduces additional complexities:

- camera motion and variable viewpoints
- occlusion of joints
- inconsistent frame rates
- lower signal-to-noise ratio
- partial body visibility

These factors may require adjustments to preprocessing parameters.

---

### 10.5 Processing Workflow

The MLB data pipeline follows the same structure:

1. extract pose from video
2. standardize skeleton representation
3. apply preprocessing and filtering
4. compute kinematic features
5. estimate derivatives
6. apply PySINDy modeling
7. evaluate and interpret results

The concrete Stage B adapter boundary should be defined in a dedicated input
contract so that baseball sources can be normalized before entering the reused
pipeline. See:

- `docs/stage_b_input_contract.md`

---

### 10.6 Expected Outcomes

Applying the pipeline to baseball data may produce:

- pitching dynamical equations
- batting/swing dynamical equations
- comparison of motion dynamics with proxy results
- preliminary biomechanical insights

---

### 10.7 Optional Implementation

This step may be completed at different levels:

#### Conceptual Implementation (Minimum)

- define pipeline adaptation strategy
- identify tools and challenges
- describe expected outputs

#### Partial Implementation (Advanced)

- process a small set of baseball video clips
- generate skeletal trajectories
- run the full pipeline on real data

---

### 10.8 Limitations

- pose estimation accuracy may limit model quality
- lack of ground truth biomechanical data
- variability in video conditions

Results should be interpreted as exploratory rather than definitive.

---

### 10.9 Step 10 Success Criteria

Step 10 is complete when:

- pipeline transfer methodology is clearly defined
- pose extraction approach is specified
- challenges and limitations are documented
- expected outputs are described

Optional implementation strengthens the project but is not required.

---
