# EC991 Topics in Macroeconomics

University of Essex, Department of Economics, Autumn Term 2026.

This repository holds the course materials: lecture slides, lab slides, Julia notebooks and code. New material is added each week. Announcements, deadlines and submissions are on Moodle.

| | |
| --- | --- |
| Lecturer | Aseem Patel, [aseem.patel@essex.ac.uk](mailto:aseem.patel@essex.ac.uk) |
| Office | 3.310A |
| Academic support hours | Wednesdays, 13:00–15:00, in person or on Zoom. Please email beforehand. |
| Module outline | [EC991_Module_Outline.pdf](syllabus/EC991_Module_Outline.pdf) |

## Schedule

| Week | Topic | Lecture | Lab |
| --- | --- | --- | --- |
| 1 | Introduction and Computing Toolkit | [Slides](week01/lecture/week01_motivation.pdf) | [Slides](week01/lab/week01_tutorial_slides.pdf) · [Setup guide](https://aseem-patel.github.io/EC991/setup/week01_tutorial.html) · [Pluto notebook](week01/lab/week01_julia_basics.jl) · [Python version](week01/lab/week01_python_basics.py) |

Later weeks appear here as they are released.

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

## AI tools

You may use AI assistants for the problem sets and in the labs, provided you can explain and defend every line you hand in. See Part 9 of the [setup guide](https://aseem-patel.github.io/EC991/setup/week01_tutorial.html#agent).

## Questions

For problems with setup or with the code in this repository, open an [issue](https://github.com/Aseem-Patel/EC991/issues). For anything about your own work, email me or come to the support hours.

## Acknowledgements

The content of this course builds on the lecture notes of Tomás R. Martinez. All errors are mine.

## Licence

Slides and documents: [CC BY-NC-SA 4.0](LICENSE). Code: [MIT](LICENSE-CODE).
