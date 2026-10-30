"""Final capstone presentation on the department template (finalCAp/Presentation_Template.pptx).

    python docs/deliverables/make_deck.py <template.pptx> <out.pptx>

Numbers come from docs/evaluation_report.md and docs/name_call_eval.md (run 20261030-185320_a_real_s3).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ASSETS = Path(__file__).parent / "assets"
RED = RGBColor(0xBE, 0x2F, 0x2C)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x8A, 0x8A, 0x8A)
LIGHT = RGBColor(0xF6, 0xE7, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TITLE_FONT, BODY_FONT = "Times New Roman", "Calibri"

TEAM = [("Raj Singh", "70022300051", "B125"), ("Prachi Teotia", "70022300162", "B132"),
        ("Advika Mishra", "70022300146", "B076"), ("Mahi Mayani", "70022300180", "B073")]
TITLE = ("SAHARA: A Real-Time, On-Device Sound Event Detection and Dual-Channel Alert System for the "
         "Deaf and Hard-of-Hearing, with India-Specific Acoustic Adaptation")
FOOTER = "Computer Engineering Department, MPSTME, Mumbai Campus"


def text(slide, x, y, w, h, paras, size=16, font=BODY_FONT, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """paras: list of str or (str, dict(bold, size, color, level, bullet))."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, p in enumerate(paras):
        s, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = o.get("align", align)
        bullet = o.get("bullet", False)
        level = o.get("level", 0)
        run = para.add_run()
        run.text = (("•  " if level == 0 else "–  ") + s) if bullet else s
        if bullet and level:
            para.level = 1
        f = run.font
        f.name = o.get("font", font)
        f.size = Pt(o.get("size", size - (2 if level else 0)))
        f.bold = o.get("bold", False)
        f.italic = o.get("italic", False)
        f.color.rgb = o.get("color", color)
        para.space_after = Pt(o.get("after", 5))
    return box


def bullets(slide, x, y, w, h, items, size=16):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            paras.append((it[0], {"bullet": True, "bold": True}))
            paras += [(sub, {"bullet": True, "level": 1}) for sub in it[1]]
        else:
            paras.append((it, {"bullet": True}))
    return text(slide, x, y, w, h, paras, size=size)


def table(slide, x, y, w, rows, col_w=None, size=11, header=True, row_h=0.3):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        t.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.fill.solid()
            head = header and r == 0
            cell.fill.fore_color.rgb = RED if head else (LIGHT if r % 2 == 0 else WHITE)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.name = BODY_FONT
                    run.font.bold = head
                    run.font.color.rgb = WHITE if head else DARK
    return t


class Deck:
    def __init__(self, template: str):
        self.prs = Presentation(template)
        self.layout = self.prs.slide_layouts[1]
        self.n = 1

    def slide(self, title: str):
        s = self.prs.slides.add_slide(self.layout)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        self.n += 1
        text(s, 3.85, 0.78, 5.95, 0.72, [(title, {"bold": True})], size=26 if len(title) <= 32 else 21, font=TITLE_FONT, color=RED,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 2.85, 6.96, 4.3, 0.35, [FOOTER], size=9, color=GREY, align=PP_ALIGN.CENTER)
        text(s, 9.0, 6.96, 0.5, 0.35, [str(self.n)], size=9, color=GREY, align=PP_ALIGN.RIGHT)
        return s

    def picture(self, s, name, x, y, w=None, h=None):
        kw = {}
        if w:
            kw["width"] = Inches(w)
        if h:
            kw["height"] = Inches(h)
        return s.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), **kw)


def title_slide(prs: Presentation) -> None:
    s = prs.slides[0]
    shapes = list(s.placeholders)
    head, body = shapes[0], shapes[1]
    tf = head.text_frame
    tf.clear()
    lines = [("Capstone Project", 30), ("Final Presentation", 26), ("A.Y. 2026-2027", 22)]
    for i, (line, size) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), True, TITLE_FONT
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Project Title: " + TITLE
    r.font.size, r.font.bold, r.font.name = Pt(15), True, TITLE_FONT
    body.text_frame.clear()
    r = body.text_frame.paragraphs[0].add_run()
    r.text = "Presented by :"
    r.font.size, r.font.name, r.font.color.rgb = Pt(16), TITLE_FONT, GREY
    t = table(s, 1.5, 4.55, 7.6, [[n for n, _, _ in TEAM], [sap for _, sap, _ in TEAM], [roll for _, _, roll in TEAM]],
              size=14, header=True, row_h=0.38)
    for r, fill in enumerate([RED, RGBColor(0x8C, 0x8C, 0x8C), RED]):
        for c in range(len(TEAM)):
            cell = t.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    run.font.color.rgb, run.font.bold = WHITE, True
    text(s, 1.33, 5.75, 7.9, 0.5, [("Under the guidance of : Prof. Priti Rumao, Computer Engineering Department",
                                    {"italic": False})], size=16, font=TITLE_FONT, color=GREY)


def build(template: str, out: str) -> None:
    d = Deck(template)
    title_slide(d.prs)

    s = d.slide("Introduction and Motivation")
    bullets(s, 0.45, 1.75, 5.4, 5.0, [
        "Over 430 million people live with disabling hearing loss (WHO, 2021); about 63 million in India (NPPCD).",
        "Everyday sounds carry safety and social information: smoke alarm, siren, doorbell, a baby crying, someone calling your name.",
        "For Deaf and Hard-of-Hearing (DHH) people these must become a visual or vibration signal.",
        "Existing aids: tested on Western data, report accuracy instead of false alarms, often need the cloud or a smartwatch.",
    ], size=18)
    stats = [("430 M", "people with disabling\nhearing loss (WHO)"), ("63 M", "people in India with\nsignificant hearing loss"),
             ("9", "alert types: 8 sounds\n+ your name being called")]
    for i, (big, small) in enumerate(stats):
        y = 1.8 + i * 1.7
        box = s.shapes.add_shape(1, Inches(6.15), Inches(y), Inches(3.45), Inches(1.45))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = RED
        text(s, 6.2, y + 0.1, 3.35, 0.7, [(big, {"bold": True})], size=30, font=TITLE_FONT, color=RED, align=PP_ALIGN.CENTER)
        text(s, 6.2, y + 0.75, 3.35, 0.65, [small], size=12, align=PP_ALIGN.CENTER)

    s = d.slide("Problem Statement and Objectives")
    text(s, 0.45, 1.7, 9.1, 1.3, [("Problem statement", {"bold": True, "color": RED}),
         "DHH users cannot hear safety-critical household sounds. Current tools are rarely tuned for false alarms, "
         "rarely tested in Indian homes, and often send audio off the device. We need a private, on-device system that "
         "alerts reliably, with few false alarms, in Indian conditions."], size=17)
    bullets(s, 0.45, 3.15, 9.1, 3.7, [
        "Detect 8 household sounds on the phone in real time: smoke alarm, siren, glass break, doorbell, knocking, dog bark, baby cry, appliance beep.",
        "Keep false alarms to about 4 per hour while weighting missed safety sounds 3x.",
        "Run fully offline: no audio stored, no Internet permission.",
        "Tell users when their name is called, and by whom.",
        "Collect and evaluate an Indian field dataset (SAHARA-IN).",
        "Deliver alerts as class-coded vibration on the phone and a low-cost wristband.",
    ], size=18)

    s = d.slide("Literature Review")
    table(s, 0.35, 1.7, 9.3, [
        ["Work", "Platform", "On-device", "False alarms/h", "Indian data", "Key finding / gap"],
        ["Bragg et al. 2016 [3]", "Phone app", "Yes", "No", "No", "DHH want personalisable alerts"],
        ["Findlater et al. 2019 [4]", "Survey (201)", "–", "–", "No", "Prefer wearable, glanceable alerts"],
        ["HomeSound 2020 [7]", "Home displays", "Partly", "No", "No", "False alerts erode trust"],
        ["SoundWatch 2020 [8]", "Smartwatch", "Yes", "No", "No", "Accuracy vs. battery trade-off"],
        ["ProtoSound 2022 [9]", "Phone", "Yes", "No", "No", "Few-shot personal sounds"],
        ["Huang et al. 2023 [11]", "Field study", "Yes", "No", "No", "Noise and confusable classes"],
        ["Ubicoustics 2018 [10]", "Devices", "Yes", "No", "No", "AudioSet embeddings transfer"],
        ["SAHARA (ours)", "Phone + band", "Yes", "Yes (3.5/h)", "Yes (329 clips)", "Budget-tuned alerts, name calls"],
    ], col_w=[1.85, 1.25, 0.95, 1.15, 1.15, 2.95], size=10, row_h=0.42)
    text(s, 0.35, 5.7, 9.3, 1.0, [("Gap: ", {"bold": True, "color": RED}),
         "no prior system tunes alerts against a false-alarm budget, evaluates on Indian recordings, and runs fully "
         "offline with name-call alerts."], size=14)

    s = d.slide("System Architecture")
    d.picture(s, "architecture.png", 0.3, 1.85, w=9.4)
    bullets(s, 0.45, 4.0, 9.1, 2.8, [
        "Native Kotlin foreground service: listens with the screen off; audio lives only in a 1 s RAM buffer.",
        "Frozen YAMNet (MobileNetV1, AudioSet) gives a 1024-d embedding + 521 class scores every 0.5 s.",
        "Trained head (256-128-8) scores 8 sounds; a per-class rule decides when to alert.",
        "Speech path: name spotter + speaker encoder (\"Raj is calling you\" / \"Someone is calling you\").",
        "Flutter UI: full-screen alert, history, settings, home calibration. Bluetooth: 4 bytes to the band.",
    ], size=16)

    s = d.slide("Datasets")
    table(s, 0.35, 1.7, 5.6, [
        ["Class", "AudioSet", "ESC-50", "FSD50K", "SAHARA-IN", "Total"],
        ["Smoke alarm*", "183", "–", "–", "20", "203"], ["Doorbell", "92", "–", "144", "58", "294"],
        ["Siren*", "34", "40", "132", "20", "226"], ["Knocking", "97", "40", "373", "67", "577"],
        ["Dog bark", "10", "40", "536", "11", "597"], ["Baby cry", "106", "40", "–", "14", "160"],
        ["Glass break*", "97", "40", "510", "20", "667"], ["Appliance beep", "261", "40", "206", "59", "566"],
        ["Background", "–", "1,760", "10,000", "60", "12,425"],
    ], col_w=[1.35, 0.85, 0.75, 0.85, 1.0, 0.8], size=10, row_h=0.36)
    bullets(s, 6.15, 1.7, 3.55, 5.1, [
        "Every label ID verified against the official vocabularies (fixed mislabelled AudioSet / FSD50K classes).",
        "70/15/15 split grouped by source recording: no leakage between splits.",
        "6,834 synthetic training mixes at -5 to 10 dB; 564 test mixes reported separately.",
        "SAHARA-IN: 329 clips from Mumbai homes and streets + 107 name-call clips.",
        "* safety-critical classes",
    ], size=13)

    s = d.slide("Method: Budget-Tuned Alerting")
    bullets(s, 0.45, 1.7, 4.6, 5.1, [
        "Safety classes (smoke, siren, glass) weighted 3x in the loss and in tuning.",
        "Alert rule per class: score ≥ threshold in k of the last 3 windows, 10 s cooldown.",
        "Tune every class's (threshold, k) on validation to maximise weighted detection subject to ≤ 4 false alarms per hour in total.",
        "Same tuning at 1, 2, 4, 8, 16 /h gives the app's 5-step sensitivity setting.",
        "Model selection on validation only; test used once.",
    ], size=15)
    d.picture(s, "budget.png", 5.15, 1.85, w=4.6)
    text(s, 5.15, 4.35, 4.6, 0.6, ["Detection vs. false-alarm budget (validation). We chose 4 per hour."], size=11, color=GREY)

    s = d.slide("Implementation")
    bullets(s, 0.45, 1.7, 4.7, 5.2, [
        ("Android app (Kotlin + Flutter)", ["Foreground service, energy gate, TFLite models", "Full-screen alert, history, settings",
                                           "No audio stored, no Internet permission"]),
        ("Name calls", ["Speech embeddings + logistic head for \"Aryan\"", "GE2E speaker encoder names enrolled family"]),
        ("Personalisation", ["Custom sounds (5-10 recordings)", "Home calibration (3 recordings per sound)"]),
        ("Haptics", ["Same 9 vibration patterns on phone and band", "ESP32 firmware compiles; band designed (Rs. 870)"]),
    ], size=13)
    d.picture(s, "haptics.png", 5.25, 1.9, w=4.45)
    text(s, 5.25, 4.4, 4.45, 0.6, ["Class-coded vibration patterns (from the firmware). Smoke alarm uses the fire-alarm \"temporal-three\" rhythm."],
         size=11, color=GREY)

    s = d.slide("Results: Detection on Held-Out Data")
    table(s, 0.35, 1.7, 4.9, [
        ["Class", "AP", "F1", "Clips alerted"],
        ["Smoke alarm*", "0.916", "0.858", "22/27"], ["Doorbell", "0.711", "0.432", "13/35"],
        ["Siren*", "0.875", "0.817", "28/34"], ["Knocking", "0.604", "0.410", "21/73"],
        ["Dog bark", "0.908", "0.847", "67/90"], ["Baby cry", "0.817", "0.712", "17/25"],
        ["Glass break*", "0.827", "0.790", "56/97"], ["Appliance beep", "0.567", "0.496", "17/78"],
        ["Mean", "0.778", "0.670", "56 %"],
    ], col_w=[1.6, 0.9, 0.9, 1.5], size=11, row_h=0.37)
    table(s, 5.45, 1.7, 4.2, [
        ["Alert rule (same model)", "Clips", "FA/h"],
        ["Threshold 0.5, 1 window", "0.77", "49.1"],
        ["Threshold 0.5, 2 of 3", "0.69", "34.1"],
        ["SAHARA tuned (4/h)", "0.54", "6.1"],
    ], col_w=[2.4, 0.8, 1.0], size=11, row_h=0.4)
    bullets(s, 5.45, 3.5, 4.2, 3.3, [
        "Public test: 0.778 mAP, 3.5 false alarms/h on 3.4 h of background.",
        "Tuning cuts false alarms 34.1 → 6.1 per hour on the same model.",
        "Safety sounds strongest; brief knocks and beeps weakest.",
    ], size=15)

    s = d.slide("Results: Indian Homes and Calibration")
    d.picture(s, "indian_cv.png", 0.3, 1.75, w=4.65)
    d.picture(s, "calibration.png", 5.05, 1.75, w=4.65)
    bullets(s, 0.45, 4.5, 9.1, 2.3, [
        "Indian recordings help a lot in homes the model has heard (0.621 → 0.822 mAP), not yet in unseen homes (0.600).",
        "So we built home calibration: record a sound 3 times; in unseen homes events alerted rise 19 % → 35 % (about +1 false alarm/h).",
        "Biggest gains for doorbells (20 % → 83 %), microwaves and baby cries.",
    ], size=15)

    s = d.slide("Results: Name Calls and On-Device Cost")
    table(s, 0.35, 1.75, 4.6, [
        ["Name call (\"Aryan\")", "ROC AUC", "EER"],
        ["vs. background scenes", "0.872", "19 %"],
        ["vs. similar names", "0.743", "32 %"],
        ["Caller named correctly", "29 / 38", ""],
        ["Unknown caller → \"Someone\"", "82 %", ""],
    ], col_w=[2.5, 1.1, 1.0], size=11, row_h=0.42)
    table(s, 5.15, 1.75, 4.5, [
        ["Samsung Galaxy S24", "Value"],
        ["Inference per window", "20-29 ms (on) / 47-49 ms (off)"],
        ["Models (YAMNet / head / name)", "15.0 / 1.2 / 8.1 MB"],
        ["Release APK", "45.8 MB"],
        ["Memory", "128.9 MB"],
        ["Sound → phone alert", "1.0-2.5 s"],
        ["App vs. pipeline", "54 / 54 same alerts"],
    ], col_w=[2.0, 2.5], size=11, row_h=0.42)
    text(s, 0.35, 5.1, 9.3, 1.5, ["Everything runs offline on the phone; the speech models are evaluated leave-one-recorder-out, so every tested voice is new."],
         size=13)

    s = d.slide("Live Demonstration")
    bullets(s, 0.45, 1.75, 9.1, 5.0, [
        "Start listening; lock the screen.",
        "Play a smoke alarm / doorbell from a second phone → full-screen alert + class-coded vibration.",
        "Someone calls \"Aryan\" → \"Raj is calling you\"; an unknown voice → \"Someone is calling you\".",
        "A sound outside the 8 classes (e.g., a tap) → quiet notice, no vibration.",
        "Settings: sensitivity steps and \"Calibrate for this home\".",
        "Wristband concept: interactive page with the 9 vibration patterns (docs/wristband_concept.html).",
    ], size=20)

    s = d.slide("Advantages, Limitations and Future Scope")
    bullets(s, 0.45, 1.7, 4.5, 5.1, [
        ("Advantages", ["Private: fully offline, no audio stored", "Measured false alarms per hour",
                        "Name calls with caller ID", "Home calibration and sensitivity steps"]),
        ("Limitations", ["Brief sounds (knock, beep) weakest", "SAHARA-IN small, few sessions",
                         "Wristband not yet built; no DHH user study yet"]),
    ], size=16)
    bullets(s, 5.15, 1.7, 4.5, 5.1, [
        ("Future scope", ["Build and measure the wristband (delay, battery)", "Fine-tune YAMNet for brief sounds",
                          "Record many more Indian households", "Field study with DHH users in Mumbai",
                          "New classes: pressure-cooker whistle, phone ringing"]),
    ], size=16)

    s = d.slide("Conclusion")
    bullets(s, 0.45, 1.75, 9.1, 5.0, [
        "SAHARA runs entirely on a phone, keeps no audio, and is tuned for false alarms per hour.",
        "Budget tuning cut false alarms more than five-fold on the same model (34.1 → 6.1 /h).",
        "Public test: 0.778 mAP at 3.5 false alarms per hour; safety sounds detected best.",
        "Indian recordings teach the model a home's own sounds; 3-recording calibration lifts unseen homes from 19 % to 35 %.",
        "Name-call alerts tell users that they are called, and by whom.",
        "Paper: submitted to ICEI 2026 (IEEE format).",
    ], size=19)

    s = d.slide("References")
    refs = [
        "[1] WHO, World Report on Hearing, 2021.",
        "[3] D. Bragg et al., A personalizable mobile sound detector app design for DHH users, ASSETS 2016.",
        "[4] L. Findlater et al., DHH preferences for wearable and mobile sound awareness, CHI 2019.",
        "[6] J. F. Gemmeke et al., Audio Set, ICASSP 2017.",
        "[7] D. Jain et al., HomeSound, CHI 2020.   [8] D. Jain et al., SoundWatch, ASSETS 2020.",
        "[9] D. Jain et al., ProtoSound, CHI 2022.   [10] G. Laput et al., Ubicoustics, UIST 2018.",
        "[11] J. Z. Huang, H. Chhabria, D. Jain, \"Not there yet\", ASSETS 2023.",
        "[19] Q. Kong et al., PANNs, IEEE/ACM TASLP 2020.   [21] M. Plakal, D. Ellis, YAMNet, 2020.",
        "[26] E. Fonseca et al., FSD50K, IEEE/ACM TASLP 2022.   [27] K. J. Piczak, ESC-50, ACM MM 2015.",
        "[28] N. Turpault et al., DESED, DCASE Workshop 2019.   [31] C. Elkan, Cost-sensitive learning, IJCAI 2001.",
        "[34] L. Wan et al., GE2E speaker verification, ICASSP 2018.   [35] ISO 8201:2017.",
        "Full list (35 references) in the paper and report.",
    ]
    text(s, 0.45, 1.7, 9.1, 5.2, refs, size=14)

    s = d.slide("Thank You")
    text(s, 0.5, 2.6, 9.0, 1.2, [("Thank you", {"bold": True})], size=40, font=TITLE_FONT, color=RED, align=PP_ALIGN.CENTER)
    text(s, 0.5, 3.8, 9.0, 1.0, ["Questions?"], size=24, font=TITLE_FONT, color=GREY, align=PP_ALIGN.CENTER)
    text(s, 0.5, 5.0, 9.0, 1.2, [" | ".join(n for n, _, _ in TEAM), "Guide: Prof. Priti Rumao"], size=14,
         color=DARK, align=PP_ALIGN.CENTER)

    d.prs.save(out)
    print(f"{out}: {len(d.prs.slides)} slides")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
