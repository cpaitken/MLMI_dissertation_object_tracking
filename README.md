# MLMI_dissertation_object_tracking
GitHub Repository for 2025 MLMI MPhil Dissertation work - Object kinematic and intent tracking using the UZH-FPV drone racing dataset

## Setup

To create a virtual environment to install dependencies:
```bash
python3 -m venv .venv
```

To activate each time:
```bash
source .venv/bin/activate
```

## Dependencies
Numpy, Matplotlib, tqdm, scipy, seaborn

## Folder Descriptions

<details>
<summary><b>Visualizers</b></summary>

Individual files to plot some of the figures within "Tracking of Erratically Moving Objects Using (non)-Gaussian Process Models". Comments describing the specific figure any particular file visualizes can be found in the first few lines.

</details>

<details>
<summary><b>Models</b></summary>

- **functions.py** -- File from Lydeard et al. (2025) to provide functionality for baseline SE model and iSE models
- **intentFunctions.py** -- File containing functions that enable intent-awareness in the g-SE and g-iSE models. KF predict and update step functions are based on those in functions.py, with alterations to account for extended-state approach. Also includes functionality for Rao-Blackwellised PF for the g-iSE model. 
- **seqInference.py** -- Not used in actual work reported in the paper; simply a first attempt at particle filtering

</details>

<details>
<summary><b>gSE</b></summary>

Files to conduct experiments for the g-SE model:
- **SEvsGSE_Tracking.py** -- Runs the SE and g-SE models on an individual trajectory (must be specified) and saves the trajectory plot and debugging information.
- **testUZH.py** -- Runs the SE and g-SE models on the entire available UZH dataset and saves the results to a folder (must be specified). Values for goal process noise variance or goal initialisation variance can be changed in lines 92 and 93. 
- **gSE_multGoalInference.py** -- Runs the goal selection task experiment for either the CSG or Quadrant trajectories dataset (must be specified)
- **clean_MultiInference.py** -- Initial debugging and testing file for the goal selection task. Not used to produce results. 

</details>

<details>
<summary><b>giSE</b></summary>

Files to conduct experiments for the g-iSE model:
- **iSEvsgiSE_Tracking.py** -- Initial file to test the g-iSE model using the basic measurement model (Section 3.2). Saves results in Debugging/iSE... and displays resulting trajectories are the same for both models.
- **giSE_knownLam_basic.py** -- Runs experiments for Section 4.3.1 on one generating convergence rate dataset and initialisation combination at a time. 
- **giSE_unknownLam_fixedLamTrajectories.py** -- Runs experiments for Section 4.3.2 on the goal-converging trajectories for one initialisation combination at a time. 
- **giSE_unknownLam_UZH.py** -- Runs experiments for Section 4.3.2 on the UZH-FPV dataset. Required specification of sigma_g, G_var, and lambda clip value if used, as well as if the goal is intiailised to the correct endpoint or 500,500.
- **giSE_unknownLam_basic.py** -- Initial debugging and testing file for estimating the lambda value using RBPF. Not used to produce results.

</details>

<details>
<summary><b>GenerationCode</b></summary>

- **generatingModel.py** -- Catch-all file used to test the generation of synthetic trajectories during the work and eventually generate the goal-converging trajectories.
- **multipleTargets.py** -- Used to generate either the CSG or Quadrant Trajectories datasets. 

</details>

<details>
<summary><b>Debugging</b></summary>

Folder to save results throughout the course of this work - Contains subfolders used for debugging, notable folders are described below:
- **conv_iSE** -- contains results for the g-iSE model with converging measurement model divided into tests using a known Lambda (constantLambda) or estimated Lambda (varyingLambda).
- **gSE_Params** -- contains results for the kernel parameter sweep of the gSE model on the UZH-FPV dataset.
- **iSE** -- first results for a g-iSE model using a basic measurement model (Section 3.2).
- **SE_MultTarget** -- results for goal selection task of g-SE model on CSG and Quadrant Trajectory Datasets, plus results if using separate state sets for tracking and goal prediction. 
- **UZH** - all results during testing of the g-SE model on the UZH Dataset, including with dynamic goal process.
- **Remaining Folders** -- UZH Debugging runs throughout the course of the work.

</details>

<details>
<summary><b>Data</b></summary>

- **UZH** - The UZH-FPV Dataset, divided into categories by difficulty
- **Generated:**
  - **CSG_Bridging** -- CSG Dataset used for goal selection task
  - **CSG_TrackParamStudy** -- Dataset used to select parameters for CSG dataset generation
  - **giSE_convergingMeasModel** -- goal-converging trajectories
  - **goalConvergingTrack_ParamStudy** -- Initial goal-converging trajectories used to decide which generating convergence rate values to include
  - **Quad_BridgingGP** - Quadrant Trajectory Dataset

</details>
        