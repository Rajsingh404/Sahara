# SAHARA paper: what is still missing

`sahara.tex` is in the conference's Springer SVProc format. Build it with `tectonic -X compile sahara.tex`
(or `pdflatex sahara.tex` twice); `svproc.cls`, `aliascnt.sty`, `remreset.sty` and `figures/` must sit in the same
folder. Regenerate the figures from the selected run with `python docs/paper/make_figures.py`.

Every number in the paper comes from the selected run folder (`models/logs/20261005-195620_b_synthetic/`),
`docs/evaluation_report.md`, `docs/name_call_eval.md` or `docs/dataset_card.md`. Every red **TBD** is something
nobody has supplied or measured yet. Don't submit while any are left.

## Measurements still to make

| Paper location | What to do | Where it comes from |
|---|---|---|
| Table "On-device cost", battery row | Phone unplugged, listening for an hour; read the battery % before and after | Phase 6 |
| Table "On-device cost", wristband rows | Build the band; time phone alert → buzz; multimeter in series for idle current; battery life = 500 mAh ÷ current, or run it flat | Phase 8, `firmware/esp32_wristband/README.md` |

If you can't measure these before submitting, delete the rows and keep the sentence in *Discussion and Limitations*
that says they are not measured.

## Details only you can fill in

- Advika's and Mahi's surnames, the guide's name (Prof. Priti Rumao or Dr. Kanikar?), the department, a corresponding e-mail
- Acknowledgement text

## Before submitting

1. Check the venue's page limit and whether review is double-blind. If it is, remove the names and the NMIMS line.
   Springer asks for an abstract of 70–150 words; ours is about 150, so trim rather than add.
2. If the model, alert settings or name-call model are retrained, regenerate `docs/evaluation_report.md` and
   `docs/name_call_eval.md`, re-run `make_figures.py`, and update the numbers in the text and tables to match.
3. Never quote 0.618, 0.651, 0.783 or 0.913: they are superseded (see the end of `docs/evaluation_report.md`).
4. The openWakeWord speech models are CC BY-NC-SA 4.0. The paper says so; keep it if the venue asks about licences.
