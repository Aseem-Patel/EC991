# EC991 Topics in Macroeconomics

University of Essex, Department of Economics, Autumn Term 2026.

This repository holds the course materials: lecture slides, lab slides, Julia notebooks and code. New material is added each week. Announcements are on Moodle; problem sets are handed in through GitHub (see [Problem sets](#problem-sets)).

| | |
| --- | --- |
| Lecturer | Aseem Patel, [aseem.patel@essex.ac.uk](mailto:aseem.patel@essex.ac.uk) |
| Office | 3.310A |
| Academic support hours | Wednesdays, 13:00–15:00, in person or on Zoom. Please email beforehand. |
| Module outline | [EC991_Module_Outline.pdf](syllabus/EC991_Module_Outline.pdf) |

## Schedule

| Week | Topic | Lecture | Lab |
| --- | --- | --- | --- |
| 1 | Lecture: Motivation – Household Heterogeneity<br>Lab: Introduction and Computing Toolkit | [Slides](week01/lecture/week01_motivation.pdf) | [Slides](week01/lab/week01_tutorial_slides.pdf) · [Setup guide](https://aseem-patel.github.io/EC991/setup/week01_tutorial.html) · [Pluto notebook](week01/lab/week01_julia_basics.jl) · [Python version](week01/lab/week01_python_basics.py) |
| 2 | Lecture: The Consumption–Savings Problem<br>Lab: Dynamic Programming and Value Function Iteration | [Slides](week02/lecture/week02_consumption_savings.pdf) | [Slides](week02/lab/week02_lab_deterministic_vfi.pdf) · [Code guide](https://aseem-patel.github.io/EC991/week02/lab/week02_code_guide.html) · [Pluto notebook](week02/lab/week02_lab.jl) · [Code package](week02/lab/EC991Week2) · Standalone notebooks: [infinite horizon (VFI)](week02/lab/week02_vfi_standalone.jl), [finite horizon (backward induction)](week02/lab/week02_backward_induction_standalone.jl) · Interactive pages: [VFI on a grid](https://aseem-patel.github.io/EC991/week02/lab/week02_vfi_on_a_grid.html), [backward induction on a grid](https://aseem-patel.github.io/EC991/week02/lab/week02_backward_induction_on_a_grid.html) |
| 3 | Lecture: The Aiyagari Model<br>Lab: Solving Aiyagari, Part 1 | [Slides](week03/lecture/week03_aiyagari_model.pdf) | [Slides](week03/lab/week03_solving_aiyagari_part1.pdf) · [Code guide](https://aseem-patel.github.io/EC991/week03/lab/week03_code_guide.html) · [Pluto notebook](week03/lab/week03_lab.jl) · [Code package](week03/lab/EC991Week3) |
| 4 | Lecture: Computing the Aiyagari Equilibrium<br>Lab: Solving Aiyagari, Part 2 | [Slides](week04/lecture/week04_aiyagari_equilibrium.pdf) | [Code guide](https://aseem-patel.github.io/EC991/week04/lab/week04_code_guide.html) · [Pluto notebook](week04/lab/week04_lab.jl) · [Code package](week04/lab/EC991Week4) |
| 5 | Lecture: Firm Dynamics: the Hopenhayn Model<br>Lab: Solving Hopenhayn | [Slides](week05/lecture/week05_hopenhayn.pdf) | [Slides](week05/lab/week05_solving_hopenhayn.pdf) · [Code guide](https://aseem-patel.github.io/EC991/week05/lab/week05_code_guide.html) · [Pluto notebook](week05/lab/week05_lab.jl) · [Code package](week05/lab/EC991Week5) |
| 6 | Lecture: Market Power and Misallocation<br>Lab: Firing Costs: Hopenhayn and Rogerson | [Slides](week06/lecture/week06_market_power.pdf) | [Slides](week06/lab/week06_hopenhayn_rogerson.pdf) · [Code guide](https://aseem-patel.github.io/EC991/week06/lab/week06_code_guide.html) · [Pluto notebook](week06/lab/week06_lab.jl) · [Code package](week06/lab/EC991Week6) · Standalone notebook: [the Atkeson and Burstein model](week06/lab/week06_atkeson_burstein.jl) |
| 7 | Lecture: Aggregate Shocks and Transition Dynamics | [Slides](week07/lecture/week07_aggregate_shocks_transitions.pdf) | |

Later weeks appear here as they are released.

## Problem sets

| Problem set | Deadline | |
| --- | --- | --- |
| 1 | Friday 18 December 2026, 12:00 (UK time) | [Problem set](problem-sets/ps1.pdf) |

You hand in your report and code through a private GitHub repository, not Moodle. You will receive an email invitation to it: accept it, then follow the repository's `README.md`. At the deadline the repository becomes read-only, and what is on GitHub then is your submission.

## Getting started

Follow the [setup guide](https://aseem-patel.github.io/EC991/setup/week01_tutorial.html) from the week 1 lab. It installs Julia, VS Code, Git and Pluto, and walks you through your first GitHub repository. In short, once Julia and Git are installed:

```bash
git clone https://github.com/Aseem-Patel/EC991.git
cd EC991
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. setup/week01_check.jl
```

Every line of the last command should print `PASS`.

To open a notebook, start Pluto from the same folder:

```bash
julia --project=. -e 'using Pluto; Pluto.run()'
```

Then paste the notebook's path, for example `week01/lab/week01_julia_basics.jl`, into the *Open a notebook* box.

## Each week

Update your copy before each lecture and lab:

```bash
cd EC991
git pull
```

Do not edit files inside this repository. Copy what you want to work on into your own repository (`ec991-work` in the setup guide) and edit it there. If you edit files here, `git pull` will run into conflicts with the next update.

## Questions

For problems with setup or with the code in this repository, open an [issue](https://github.com/Aseem-Patel/EC991/issues). For anything about your own work, email me or come to the support hours.

## Acknowledgements

The content of this course builds on the lecture notes of [Tomás R. Martinez](https://tomasrm.github.io/). All errors are mine.

## Licence

Slides and documents: [CC BY-NC-SA 4.0](LICENSE). Code: [MIT](LICENSE-CODE).
