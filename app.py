import customtkinter as ctk
import joblib
import numpy as np
import os
import pandas as pd
import keras

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
APP_BG   = "#C8D8D8"   # medium teal-gray — clearly not white
CARD_BG  = "#FFFFFF"
HDR_BG   = "#1B2A4A"
TEAL     = "#008B8B"
TEAL_H   = "#009C9C"
TEAL_LT  = "#B8D8D8"
T_DARK   = "#1B2A4A"
T_MED    = "#2E3D5A"
T_LIGHT  = "#556070"
BORDER   = "#A8C0C0"
GREEN    = "#17A84A"
GREEN_BG = "#C8F0D8"
RED      = "#D42020"
RED_BG   = "#FFD0D0"
AMBER    = "#C07800"
AMBER_BG = "#FFE8B0"
PROG_BG  = "#A8C8C8"

ADMIT_THRESHOLD = 0.30

# Font set after window creation so tkfont.families() works
FONT_FAMILY = "Segoe UI"

def fnt(s, bold=False):
    return ctk.CTkFont(family=FONT_FAMILY, size=s,
                       weight="bold" if bold else "normal")

# ══════════════════════════════════════════════════════════════════════════════
#  DEPARTMENT ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def assign_department(cd):
    scores = {
        "Cardiology": (
            cd.get("cc_chestpain",0)*3 + cd.get("cc_palpitations",0)*2 +
            cd.get("cc_irregularheartbeat",0)*2 + cd.get("cc_tachycardia",0)*2 +
            cd.get("cc_chesttightness",0)*2 + cd.get("cc_hypertension",0) +
            cd.get("cc_hypotension",0)*2 + cd.get("cc_syncope",0)
        ),
        "Neurology": (
            cd.get("cc_headache",0)*2 + cd.get("cc_seizures",0)*3 +
            cd.get("cc_numbness",0)*2 + cd.get("cc_strokealert",0)*3 +
            cd.get("cc_dizziness",0)*2 + cd.get("cc_lossofconsciousness",0)*3 +
            cd.get("cc_migraine",0)*2 + cd.get("cc_alteredmentalstatus",0)*2
        ),
        "Pulmonology": (
            cd.get("cc_shortnessofbreath",0)*3 + cd.get("cc_breathingdifficulty",0)*3 +
            cd.get("cc_wheezing",0)*2 + cd.get("cc_respiratorydistress",0)*3 +
            cd.get("cc_cough",0) + cd.get("cc_dyspnea",0)*3
        ),
        "Gastroenterology": (
            cd.get("cc_abdominalpain",0)*2 + cd.get("cc_nausea",0) +
            cd.get("cc_gibleeding",0)*3 + cd.get("cc_diarrhea",0) +
            cd.get("cc_emesis",0) + cd.get("cc_epigastricpain",0)*2 +
            cd.get("cc_constipation",0)
        ),
        "Orthopedics": (
            cd.get("cc_backpain",0)*2 + cd.get("cc_jointinjury",0)*2 +
            cd.get("cc_shoulderpain",0)*2 + cd.get("cc_kneepain",0)*2 +
            cd.get("cc_hippain",0)*2 + cd.get("cc_neckpain",0) +
            cd.get("cc_legpain",0) + cd.get("cc_arminjury",0)*2 +
            cd.get("cc_footinjury",0)*2
        ),
        "Dermatology": (
            cd.get("cc_rash",0)*2 + cd.get("cc_skinproblem",0)*2 +
            cd.get("cc_burn",0)*2 + cd.get("cc_cellulitis",0)*2 +
            cd.get("cc_abscess",0)*2
        ),
        "Psychiatry": (
            cd.get("cc_suicidal",0)*3 + cd.get("cc_anxiety",0) +
            cd.get("cc_depression",0)*2 + cd.get("cc_hallucinations",0)*3 +
            cd.get("cc_psychoticsymptoms",0)*3 + cd.get("cc_agitation",0)*2
        ),
        "Urology": (
            cd.get("cc_urinarytractinfection",0)*2 + cd.get("cc_hematuria",0)*2 +
            cd.get("cc_flankpain",0)
        ),
        "Ophthalmology": (
            cd.get("cc_eyepain",0)*2 + cd.get("cc_blurredvision",0)*2
        ),
        "ENT": (
            cd.get("cc_earpain",0)*2 + cd.get("cc_sorethroat",0) +
            cd.get("cc_epistaxis",0)*2
        ),
        "Emergency Surgery": (
            cd.get("cc_trauma",0)*3 + cd.get("cc_fulltrauma",0)*3 +
            cd.get("cc_motorvehiclecrash",0)*3 + cd.get("cc_headinjury",0)*2 +
            cd.get("cc_assaultvictim",0)*2 + cd.get("cc_laceration",0)
        ),
        "Toxicology": (
            cd.get("cc_overdose-accidental",0)*3 +
            cd.get("cc_overdose-intentional",0)*3 +
            cd.get("cc_poisoning",0)*3 + cd.get("cc_alcoholintoxication",0)*2
        ),
        "Infectious Disease": (
            cd.get("cc_fever",0)*2 + cd.get("cc_influenza",0)*2 +
            cd.get("cc_uri",0)
        ),
        "Endocrinology": (
            cd.get("cc_hyperglycemia",0)*2 +
            cd.get("cc_decreasedbloodsugar-symptomatic",0)*2
        ),
    }
    best = max(scores, key=scores.get)
    return "General ER" if scores[best] == 0 else best


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD MAPS
# ══════════════════════════════════════════════════════════════════════════════
COMPLAINT_KEYWORDS = {
    "cc_chestpain":       ["cardiac arrest","heart stopped","no pulse","chest pain",
                           "chest ache","chest discomfort","chest pressure","chest tight"],
    "cc_shortnessofbreath": ["short of breath","shortness of breath","cant breathe",
                              "can't breathe","difficulty breathing","breathless",
                              "cannot breathe"],
    "cc_headache":        ["headache","head pain","head ache"],
    "cc_unresponsive":    ["unresponsive","unconscious","not responding","no response"],
    "cc_migraine":        ["migraine"],
    "cc_respiratorydistress": ["respiratory distress","not breathing"],
    "cc_dizziness":       ["dizzy","dizziness","lightheaded","light headed"],
    "cc_nausea":          ["nausea","nauseous","feel sick"],
    "cc_emesis":          ["vomit","vomiting","threw up","throwing up"],
    "cc_fever":           ["fever","high temperature","burning up"],
    "cc_abdominalpain":   ["abdominal pain","stomach pain","belly pain","stomach ache"],
    "cc_backpain":        ["back pain","back ache","lower back"],
    "cc_chills":          ["chills","shivering","shaking"],
    "cc_cough":           ["cough","coughing"],
    "cc_rash":            ["rash","skin rash","hives"],
    "cc_seizures":        ["seizure","convulsion","fit"],
    "cc_numbness":        ["numb","numbness","tingling"],
    "cc_palpitations":    ["palpitation","heart racing","heart pounding"],
    "cc_syncope":         ["fainted","faint","passed out","blackout"],
    "cc_weakness":        ["weakness","weak","no strength"],
    "cc_trauma":          ["trauma","accident","injury","hurt"],
    "cc_laceration":      ["cut","laceration","bleeding wound"],
    "cc_burn":            ["burn","burned","burning skin"],
    "cc_eyepain":         ["eye pain","eye hurts","painful eye"],
    "cc_earpain":         ["ear pain","ear ache","earache"],
    "cc_sorethroat":      ["sore throat","throat pain","throat hurts"],
    "cc_urinarytractinfection": ["uti","urinary","burning urination"],
    "cc_depression":      ["depressed","depression","hopeless"],
    "cc_suicidal":        ["suicidal","want to die","kill myself"],
    "cc_anxiety":         ["anxiety","anxious","panic attack","panic"],
    "cc_irregularheartbeat": ["irregular heartbeat","arrhythmia"],
    "cc_hypertension":    ["high blood pressure","hypertension"],
    "cc_hypotension":     ["low blood pressure","hypotension"],
    "cc_overdose-accidental":  ["overdose","took too much"],
    "cc_overdose-intentional": ["intentional overdose","took pills on purpose"],
    "cc_alcoholintoxication":  ["drunk","alcohol","intoxicated"],
    "cc_motorvehiclecrash": ["car accident","car crash","road accident"],
    "cc_headinjury":      ["head injury","hit head","head trauma"],
    "cc_kneepain":        ["knee pain","knee hurts","knee injury"],
    "cc_shoulderpain":    ["shoulder pain","shoulder hurts"],
    "cc_hippain":         ["hip pain","hip hurts"],
    "cc_legpain":         ["leg pain","leg hurts"],
    "cc_arminjury":       ["arm injury","arm pain","arm hurts"],
    "cc_footinjury":      ["foot injury","foot pain","foot hurts"],
    "cc_blurredvision":   ["blurred vision","blurry vision"],
    "cc_tachycardia":     ["fast heart rate","tachycardia"],
    "cc_dyspnea":         ["dyspnea","breathless"],
    "cc_alteredmentalstatus": ["confused","confusion","disoriented"],
    "cc_lossofconsciousness": ["lost consciousness","unconscious"],
    "cc_strokealert":     ["stroke","face drooping"],
    "cc_gibleeding":      ["blood in stool","rectal bleeding","bloody stool"],
    "cc_hematuria":       ["blood in urine","bloody urine"],
    "cc_epistaxis":       ["nosebleed","nose bleeding"],
    "cc_hyperglycemia":   ["high blood sugar","hyperglycemia"],
    "cc_decreasedbloodsugar-symptomatic": ["low blood sugar","hypoglycemia"],
    "cc_influenza":       ["flu","influenza"],
    "cc_uri":             ["cold","runny nose","congestion"],
    "cc_cellulitis":      ["cellulitis","skin infection"],
    "cc_abscess":         ["abscess","boil","pus"],
    "cc_diarrhea":        ["diarrhea","loose stool"],
    "cc_constipation":    ["constipation","cant poop"],
    "cc_poisoning":       ["poisoning","swallowed poison"],
    "cc_hallucinations":  ["hallucination","seeing things"],
    "cc_agitation":       ["agitated","aggressive","violent"],
    "cc_flankpain":       ["flank pain","side pain","kidney pain"],
    "cc_wheezing":        ["wheeze","wheezing"],
    "cc_breathingdifficulty": ["breathing difficulty","trouble breathing"],
    "cc_chesttightness":  ["chest tightness","tight chest"],
    "cc_fulltrauma":      ["major trauma","severe accident"],
    "cc_assaultvictim":   ["assault","attacked","beaten","stabbed","shot"],
    "cc_neckpain":        ["neck pain","stiff neck"],
    "cc_jointinjury":     ["joint injury","joint pain"],
    "cc_skinproblem":     ["skin problem","itching","itchy"],
    "cc_psychoticsymptoms": ["psychosis","psychotic","paranoid"],
    "cc_epigastricpain":  ["epigastric pain","upper stomach pain"],
}

HISTORY_KEYWORDS = {
    "diabmelnoc":    ["diabetes","diabetic"],
    "htn":           ["hypertension","high blood pressure"],
    "asthma":        ["asthma","asthmatic"],
    "chfnonhp":      ["heart failure","congestive heart failure"],
    "coronathero":   ["coronary artery disease","heart disease","cad"],
    "copd":          ["copd","emphysema","chronic bronchitis"],
    "dysrhythmia":   ["arrhythmia","atrial fibrillation","afib"],
    "epilepsycnv":   ["epilepsy","seizure disorder"],
    "osteoarthros":  ["osteoarthritis","arthritis"],
    "backproblem":   ["back problems","chronic back pain"],
    "thyroiddsor":   ["thyroid","hypothyroid","hyperthyroid"],
    "rheumarth":     ["rheumatoid arthritis"],
    "anxietydisorders":          ["anxiety disorder","panic disorder"],
    "mooddisorders":             ["depression","bipolar","mood disorder"],
    "alcoholrelateddisorders":   ["alcoholism","alcohol abuse"],
    "substancerelateddisorders": ["drug abuse","substance abuse"],
    "chrkidneydisease":          ["kidney disease","renal disease","ckd"],
    "hyperlipidem":              ["high cholesterol","hyperlipidemia"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class ERApp:

    def __init__(self, root: ctk.CTk):
        self.root = root

        # Detect Poppins now that the Tk root exists
        import tkinter.font as tkfont
        global FONT_FAMILY
        FONT_FAMILY = "Poppins" if "Poppins" in tkfont.families() else "Segoe UI"

        root.title("AI ER Hospital Receptionist")
        root.geometry("1400x900")
        root.minsize(1200, 800)
        root.configure(fg_color=APP_BG)

        self.pipeline_a    = None
        self.model_b       = None
        self.scaler_b      = None
        self.le_gender     = None
        self.feature_cols  = None
        self.models_loaded = False
        self._result_widgets: list = []

        self._load_models()
        self._build_ui()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            self.pipeline_a   = joblib.load(os.path.join(d, "model_a_pipeline.pkl"))
            self.model_b      = keras.models.load_model(os.path.join(d, "model_b_nn.keras"))
            self.scaler_b     = joblib.load(os.path.join(d, "model_b_scaler.pkl"))
            self.le_gender    = joblib.load(os.path.join(d, "le_gender.pkl"))
            self.feature_cols = joblib.load(os.path.join(d, "feature_cols.pkl"))
            self.models_loaded = True
        except Exception as e:
            self._load_err = str(e)

    # ── Top-level layout ───────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_body()

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self.root, fg_color=HDR_BG, height=76, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left", padx=30, fill="y")

        ctk.CTkLabel(left, text="✚", fg_color=TEAL, text_color="white",
                     corner_radius=24, width=48, height=48,
                     font=fnt(22, True)).pack(side="left", pady=14)

        titles = ctk.CTkFrame(left, fg_color="transparent")
        titles.pack(side="left", padx=14, fill="y")
        ctk.CTkLabel(titles, text="AI ER Hospital Receptionist",
                     text_color="white", fg_color="transparent",
                     font=fnt(18, True)).pack(anchor="w", pady=(18, 0))
        ctk.CTkLabel(titles, text="Intelligent Triage & Patient Assessment System",
                     text_color="#7A8FA8", fg_color="transparent",
                     font=fnt(10)).pack(anchor="w")

        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right", padx=30, fill="y")

        ctk.CTkButton(right, text="+ New Assessment",
                      fg_color=TEAL, hover_color=TEAL_H,
                      text_color="white", corner_radius=10,
                      font=fnt(11, True), height=40, width=170,
                      command=self._clear).pack(side="right", pady=18)

        dot_color = TEAL if self.models_loaded else RED
        ctk.CTkLabel(right, text="●", text_color=dot_color,
                     fg_color="transparent",
                     font=fnt(14)).pack(side="right", padx=(0, 14), pady=18)

    # ── Body ───────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self.root, fg_color=APP_BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=30, pady=22)
        body.columnconfigure(0, weight=48, uniform="col")
        body.columnconfigure(1, weight=52, uniform="col")
        body.rowconfigure(0, weight=1)

        # Left panel
        lp = ctk.CTkFrame(body, fg_color=APP_BG, corner_radius=0)
        lp.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        lp.columnconfigure(0, weight=1)
        lp.rowconfigure(1, weight=1)
        self._section_header(lp, "👤", "Patient Inputs", row=0)

        input_card = ctk.CTkFrame(lp, fg_color=CARD_BG, corner_radius=18,
                                   border_width=1, border_color=BORDER)
        input_card.grid(row=1, column=0, sticky="nsew")
        input_card.columnconfigure(0, weight=1)
        input_card.rowconfigure(0, weight=1)
        input_card.rowconfigure(1, weight=0)
        self._build_input_form(input_card)

        # Right panel
        rp = ctk.CTkFrame(body, fg_color=APP_BG, corner_radius=0)
        rp.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        rp.columnconfigure(0, weight=1)
        rp.rowconfigure(1, weight=1)
        self._section_header(rp, "🧠", "AI Assessment & Output", row=0)

        self._out_scroll = ctk.CTkScrollableFrame(
            rp, fg_color=APP_BG, corner_radius=0, border_width=0,
            scrollbar_fg_color=APP_BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=TEAL,
        )
        self._out_scroll.grid(row=1, column=0, sticky="nsew")
        self._out_scroll.columnconfigure(0, weight=1)
        self._build_idle_output()

    def _section_header(self, parent, icon, title, row):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(f, text=icon, fg_color=TEAL, text_color="white",
                     corner_radius=17, width=34, height=34,
                     font=fnt(15)).pack(side="left")
        ctk.CTkLabel(f, text=f"  {title}", text_color=TEAL,
                     fg_color="transparent",
                     font=fnt(16, True)).pack(side="left")

    # ── Input form ─────────────────────────────────────────────────────────────
    def _build_input_form(self, card):
        scroll = ctk.CTkScrollableFrame(
            card, fg_color=CARD_BG, corner_radius=0, border_width=0,
            scrollbar_fg_color=CARD_BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=TEAL,
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=(20, 0))
        scroll.columnconfigure(0, weight=1)

        # Age + Gender
        r1 = ctk.CTkFrame(scroll, fg_color=CARD_BG)
        r1.pack(fill="x", pady=(0, 14))
        r1.columnconfigure(0, weight=1)
        r1.columnconfigure(1, weight=1)
        age_col = ctk.CTkFrame(r1, fg_color=CARD_BG)
        age_col.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._lbl(age_col, "Age")
        self.v_age = ctk.StringVar(value="45")
        self._entry(age_col, self.v_age)
        gen_col = ctk.CTkFrame(r1, fg_color=CARD_BG)
        gen_col.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._lbl(gen_col, "Gender")
        self.v_gender = ctk.StringVar(value="Male")
        ctk.CTkComboBox(gen_col, variable=self.v_gender,
                        values=["Male", "Female"],
                        fg_color=CARD_BG, border_color=BORDER, border_width=1,
                        text_color=T_DARK, button_color=BORDER,
                        button_hover_color=TEAL_LT,
                        dropdown_fg_color=CARD_BG, dropdown_text_color=T_DARK,
                        dropdown_hover_color=TEAL_LT,
                        font=fnt(12), height=44,
                        state="readonly").pack(fill="x")

        # HR / DBP / SBP
        r2 = ctk.CTkFrame(scroll, fg_color=CARD_BG)
        r2.pack(fill="x", pady=(0, 14))
        for i in range(3):
            r2.columnconfigure(i, weight=1)
        for i, (lbl, attr, val) in enumerate([
            ("HR (bpm)", "v_hr", "88"),
            ("BP – Diastolic (mmHg)", "v_dbp", "80"),
            ("BP – Systolic (mmHg)", "v_sbp", "120"),
        ]):
            c = ctk.CTkFrame(r2, fg_color=CARD_BG)
            c.grid(row=0, column=i, sticky="ew", padx=(0, 8) if i < 2 else 0)
            self._lbl(c, lbl)
            setattr(self, attr, ctk.StringVar(value=val))
            self._entry(c, getattr(self, attr))

        # RR / Temp
        r3 = ctk.CTkFrame(scroll, fg_color=CARD_BG)
        r3.pack(fill="x", pady=(0, 14))
        r3.columnconfigure(0, weight=1)
        r3.columnconfigure(1, weight=1)
        for i, (lbl, attr, val) in enumerate([
            ("RR (breaths/min)", "v_rr", "16"),
            ("Temp (°C)", "v_temp", "37.2"),
        ]):
            c = ctk.CTkFrame(r3, fg_color=CARD_BG)
            c.grid(row=0, column=i, sticky="ew", padx=(0, 8) if i == 0 else 0)
            self._lbl(c, lbl)
            setattr(self, attr, ctk.StringVar(value=val))
            self._entry(c, getattr(self, attr))

        # Text areas
        for lbl, attr, hint in [
            ("Main Cause of Pain",  "t_complaint",
             "e.g. chest pain, seizure, shortness of breath…"),
            ("Additional Symptoms", "t_symptoms",
             "e.g. dizziness, nausea, sweating…"),
            ("Medical History",     "t_history",
             "Known conditions — optional"),
        ]:
            self._lbl(scroll, lbl)
            tb = ctk.CTkTextbox(scroll, height=64,
                                fg_color=CARD_BG, border_color=BORDER,
                                border_width=1, text_color=T_DARK,
                                font=fnt(12), activate_scrollbars=False)
            tb.pack(fill="x")
            setattr(self, attr, tb)
            ctk.CTkLabel(scroll, text=hint, text_color=T_LIGHT,
                         fg_color="transparent",
                         font=fnt(10)).pack(anchor="w", pady=(3, 12))

        # Analyze button — always visible outside scroll
        self.run_btn = ctk.CTkButton(
            card, text="Analyze Patient  ↯",
            fg_color=TEAL, hover_color=TEAL_H,
            text_color="white", corner_radius=14,
            height=55, font=fnt(14, True),
            command=self._analyze,
        )
        self.run_btn.grid(row=1, column=0, sticky="ew", padx=24, pady=(12, 22))

    def _lbl(self, parent, text):
        ctk.CTkLabel(parent, text=text, text_color=T_MED,
                     fg_color="transparent",
                     font=fnt(11, True)).pack(anchor="w", pady=(0, 5))

    def _entry(self, parent, var):
        ctk.CTkEntry(parent, textvariable=var,
                     fg_color=CARD_BG, border_color=BORDER,
                     border_width=1, text_color=T_DARK,
                     font=fnt(12), height=44).pack(fill="x")

    # ── Output helpers ─────────────────────────────────────────────────────────
    def _clear_output(self):
        for w in self._result_widgets:
            if w.winfo_exists():
                w.destroy()
        self._result_widgets.clear()

    def _track(self, w):
        self._result_widgets.append(w)
        return w

    def _card(self, parent, bg=CARD_BG, radius=18):
        c = ctk.CTkFrame(parent, fg_color=bg, corner_radius=radius,
                         border_width=1, border_color=BORDER)
        return self._track(c)

    # ── Idle ───────────────────────────────────────────────────────────────────
    def _build_idle_output(self):
        self._clear_output()
        idle = self._card(self._out_scroll)
        idle.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(idle,
                     text="Run an assessment to see AI results here.",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(13)).pack(pady=70, padx=24)

    # ── Results — all fields from the original app ─────────────────────────────
    def _build_results_output(self, priority, pri_conf, dept,
                               disp, disp_conf, admit_prob, detected):
        self._clear_output()
        sc = self._out_scroll

        is_critical = priority == "Needs-Immediate-Attention"
        is_admit    = disp == "Admit"

        # ── 1. PRIORITY — MODEL A (full-width) ────────────────────────────────
        pri_bg    = RED_BG   if is_critical else GREEN_BG
        pri_color = RED      if is_critical else GREEN
        pri_text  = "NEEDS IMMEDIATE ATTENTION" if is_critical else "CAN WAIT"

        pri_card = self._card(sc, bg=pri_bg)
        pri_card.pack(fill="x", pady=(4, 10))
        ctk.CTkLabel(pri_card, text="PRIORITY  —  MODEL A",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(11, True)).pack(anchor="w", padx=20, pady=(16, 4))
        ctk.CTkLabel(pri_card, text=pri_text,
                     text_color=pri_color, fg_color="transparent",
                     font=fnt(26, True)).pack(anchor="w", padx=20, pady=(0, 4))
        ctk.CTkLabel(pri_card, text=f"Model confidence   {pri_conf}%",
                     text_color=T_MED, fg_color="transparent",
                     font=fnt(12)).pack(anchor="w", padx=20, pady=(0, 16))

        # ── 2. DEPARTMENT | LIKELY OUTCOME (two columns) ─────────────────────
        row2 = self._track(ctk.CTkFrame(sc, fg_color="transparent"))
        row2.pack(fill="x", pady=(0, 10))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        dept_card = ctk.CTkFrame(row2, fg_color=CARD_BG, corner_radius=18,
                                  border_width=1, border_color=BORDER)
        dept_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(dept_card, text="DEPARTMENT  —  ROUTING",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(11, True)).pack(anchor="w", padx=20, pady=(16, 6))
        ctk.CTkLabel(dept_card, text=dept,
                     text_color=TEAL, fg_color="transparent",
                     font=fnt(22, True)).pack(anchor="w", padx=20, pady=(0, 16))

        out_color = AMBER if is_admit else GREEN
        out_bg    = AMBER_BG if is_admit else GREEN_BG
        out_text  = "ADMIT" if is_admit else "DISCHARGE"

        out_card = ctk.CTkFrame(row2, fg_color=out_bg, corner_radius=18,
                                 border_width=1, border_color=BORDER)
        out_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ctk.CTkLabel(out_card, text="LIKELY OUTCOME  —  MODEL B",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(11, True)).pack(anchor="w", padx=20, pady=(16, 6))
        ctk.CTkLabel(out_card, text=out_text,
                     text_color=out_color, fg_color="transparent",
                     font=fnt(26, True)).pack(anchor="w", padx=20, pady=(0, 4))
        ctk.CTkLabel(out_card, text=f"Confidence  {disp_conf}%",
                     text_color=T_MED, fg_color="transparent",
                     font=fnt(12)).pack(anchor="w", padx=20, pady=(0, 16))

        # ── 3. ADMISSION PROBABILITY ──────────────────────────────────────────
        prob_card = self._card(sc)
        prob_card.pack(fill="x", pady=(0, 10))

        ph = ctk.CTkFrame(prob_card, fg_color="transparent")
        ph.pack(fill="x", padx=20, pady=(16, 6))
        ctk.CTkLabel(ph, text="ADMISSION PROBABILITY",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(11, True)).pack(side="left")
        pct = round(admit_prob * 100, 1)
        bar_color = RED if admit_prob >= 0.65 else \
                    AMBER if admit_prob >= ADMIT_THRESHOLD else GREEN
        ctk.CTkLabel(ph, text=f"{pct}%",
                     text_color=bar_color, fg_color="transparent",
                     font=fnt(14, True)).pack(side="right")

        bar = ctk.CTkProgressBar(prob_card, fg_color=PROG_BG,
                                  progress_color=bar_color,
                                  corner_radius=5, height=12)
        bar.pack(fill="x", padx=20, pady=(0, 6))
        bar.set(admit_prob)

        foot = ctk.CTkFrame(prob_card, fg_color="transparent")
        foot.pack(fill="x", padx=20, pady=(0, 16))
        ctk.CTkLabel(foot, text="0%", text_color=T_LIGHT,
                     fg_color="transparent", font=fnt(10)).pack(side="left")
        ctk.CTkLabel(foot, text=f"Admit threshold: {int(ADMIT_THRESHOLD*100)}%",
                     text_color=T_LIGHT, fg_color="transparent",
                     font=fnt(10)).pack(side="right")

        # ── 4. DETECTED SYMPTOMS ──────────────────────────────────────────────
        if detected:
            sym_card = self._card(sc)
            sym_card.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(sym_card, text="DETECTED SYMPTOMS",
                         text_color=T_LIGHT, fg_color="transparent",
                         font=fnt(11, True)).pack(anchor="w", padx=20, pady=(16, 10))
            tags_row = ctk.CTkFrame(sym_card, fg_color="transparent")
            tags_row.pack(fill="x", padx=20, pady=(0, 16))
            names = [k.replace("cc_", "").replace("-", " ").replace("_", " ").title()
                     for k in list(detected.keys())]
            for name in names:
                ctk.CTkButton(tags_row, text=name,
                              fg_color=TEAL_LT, hover_color=TEAL_LT,
                              text_color=T_DARK, corner_radius=999,
                              height=32, font=fnt(11, True),
                              border_width=0, command=None
                              ).pack(side="left", padx=(0, 8), pady=(0, 4))

        # ── 5. BED PREPARATION BANNER ─────────────────────────────────────────
        if is_admit:
            bed = self._card(sc, bg=AMBER_BG)
            bed.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(bed,
                         text="🛏   Bed preparation recommended — admission is likely.",
                         text_color=AMBER, fg_color="transparent",
                         font=fnt(13, True)).pack(padx=20, pady=14, anchor="w")
        else:
            bed = self._card(sc, bg=GREEN_BG)
            bed.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(bed,
                         text="✓   No bed preparation needed — discharge is likely.",
                         text_color=GREEN, fg_color="transparent",
                         font=fnt(13, True)).pack(padx=20, pady=14, anchor="w")

    # ── Analysis ───────────────────────────────────────────────────────────────
    def _analyze(self):
        if not self.models_loaded:
            self._show_error("Models not found in ./models/ folder.")
            return

        complaint = self.t_complaint.get("1.0", "end-1c").strip()
        if not complaint:
            self._show_error("Please enter the patient's main complaint.")
            return

        self.run_btn.configure(text="Analyzing…", state="disabled")
        self.root.update()

        try:
            symptoms  = self.t_symptoms.get("1.0", "end-1c")
            history   = self.t_history.get("1.0", "end-1c")
            age       = float(self.v_age.get()    or 35)
            gender    = self.v_gender.get()
            hr        = float(self.v_hr.get()     or 72)
            sbp       = float(self.v_sbp.get()    or 120)
            dbp       = float(self.v_dbp.get()    or 80)
            rr        = float(self.v_rr.get()     or 16)
            temp      = float(self.v_temp.get()   or 37.0)

            all_text  = (complaint + " " + symptoms).lower()
            hist_text = history.lower()

            fv = {c: 0 for c in self.feature_cols}
            detected: dict = {}

            for cc, kws in COMPLAINT_KEYWORDS.items():
                for kw in kws:
                    if kw in all_text:
                        fv[cc] = 1
                        detected[cc] = 1
                        break

            for hc, kws in HISTORY_KEYWORDS.items():
                for kw in kws:
                    if kw in hist_text:
                        fv[hc] = 1
                        break

            try:
                fv["gender_encoded"] = self.le_gender.transform([gender])[0]
            except Exception:
                fv["gender_encoded"] = 0

            fv["age"]               = age
            fv["triage_vital_hr"]   = hr
            fv["triage_vital_sbp"]  = sbp
            fv["triage_vital_dbp"]  = dbp
            fv["triage_vital_rr"]   = rr
            fv["triage_vital_temp"] = temp

            X = pd.DataFrame([[fv.get(c, 0) for c in self.feature_cols]],
                             columns=self.feature_cols)

            priority  = self.pipeline_a.predict(X)[0]
            pri_probs = self.pipeline_a.predict_proba(X)[0]
            pri_conf  = round(float(max(pri_probs)) * 100, 1)

            X_sc        = self.scaler_b.transform(X)
            admit_prob  = float(self.model_b(X_sc, training=False).numpy()[0][0])
            disposition = "Admit" if admit_prob >= ADMIT_THRESHOLD else "Discharge"
            disp_conf   = round((admit_prob if disposition == "Admit"
                                 else 1 - admit_prob) * 100, 1)

            department = assign_department(detected)

            self._build_results_output(
                priority, pri_conf, department,
                disposition, disp_conf, admit_prob, detected
            )

        except Exception as exc:
            import traceback; traceback.print_exc()
            self._show_error(str(exc))
        finally:
            self.run_btn.configure(text="Analyze Patient  ↯", state="normal")

    def _show_error(self, msg: str):
        self._clear_output()
        err = self._card(self._out_scroll, bg=RED_BG)
        err.pack(fill="x", pady=20)
        ctk.CTkLabel(err, text=msg, text_color=RED,
                     fg_color="transparent", font=fnt(12),
                     wraplength=440, justify="center").pack(pady=40, padx=24)

    def _clear(self):
        for attr in ("t_complaint", "t_symptoms", "t_history"):
            getattr(self, attr).delete("1.0", "end")
        self.v_age.set("45");    self.v_gender.set("Male")
        self.v_hr.set("88");     self.v_sbp.set("120")
        self.v_dbp.set("80");    self.v_rr.set("16")
        self.v_temp.set("37.2")
        self._build_idle_output()


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ctk.CTk()
    ERApp(app)
    app.mainloop()
