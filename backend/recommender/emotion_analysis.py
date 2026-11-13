"""Conservative, explainable emotion inference from retrieved lyric text.

This is a multilingual keyword baseline, not a trained model. A locally
configured fine-tuned XLM-RoBERTa checkpoint can replace it; no weights are
downloaded by this module. Lyrics are processed in memory and never persisted.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Mapping


EMOTION_MODEL_VERSION = "multilingual-lyrics-lexicon-v1"
EMOTION_LABELS = ("Sad", "Happy", "Romantic", "Calm", "Energetic", "Motivated", "Nostalgic")

# Each cue list is intentionally small and high precision. Native-script and
# romanized cues are both accepted. A single cue is never sufficient evidence.
_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "English": {
        "Sad": ("heartbreak", "heartbroken", "lonely", "tears", "grief", "sorrow", "goodbye forever", "miss you", "dard", "udaas"),
        "Happy": ("laughter", "smiling", "joyful", "celebrate", "sunshine", "khushi", "santosham"),
        "Romantic": ("i love you", "my beloved", "darling", "devotion", "cherish", "pyaar", "ishq", "prema", "kadhal"),
        "Calm": ("peaceful", "stillness", "breathe slowly", "gentle night", "sukoon", "shanti", "amaithi"),
        "Energetic": ("dance all night", "feel the rhythm", "move your body", "let's go", "josh", "utsaham"),
        "Motivated": ("rise again", "never give up", "keep fighting", "believe in yourself", "himmat", "dhairyam"),
        "Nostalgic": ("those old days", "childhood memories", "remember when", "back in time", "yaadein", "gnapakam"),
    },
    "Hindi": {
        "Sad": ("दर्द", "दुख", "आँसू", "अकेला", "तन्हा", "बिछड़", "जुदाई", "udaas", "dard", "tanha", "judai", "aansu"),
        "Happy": ("खुशी", "हँसी", "मुस्कान", "जश्न", "प्रसन्न", "khushi", "hansi", "muskaan"),
        "Romantic": ("प्यार", "इश्क", "मोहब्बत", "चाहत", "सनम", "pyaar", "ishq", "mohabbat", "chahat"),
        "Calm": ("शांति", "सुकून", "आराम", "सुकून", "shaanti", "sukoon", "aaraam"),
        "Energetic": ("जोश", "नाच", "धूम", "ऊर्जा", "josh", "naach", "dhoom"),
        "Motivated": ("हौसला", "जीत", "सपने", "मंज़िल", "हिम्मत", "hausla", "jeet", "sapne", "manzil"),
        "Nostalgic": ("यादें", "पुराने दिन", "बचपन", "बीते पल", "yaadein", "purane din", "bachpan"),
    },
    "Telugu": {
        "Sad": ("బాధ", "దుఃఖం", "కన్నీరు", "ఒంటరి", "విరహం", "విడిపోవు", "badha", "duhkham", "kanniru", "ontari", "viraham"),
        "Happy": ("సంతోషం", "నవ్వు", "ఆనందం", "సంతోషంగా", "santhosham", "navvu", "anandam"),
        "Romantic": ("ప్రేమ", "ప్రియ", "మనసు", "ప్రియతమ", "prema", "priya", "manasu", "priyatama"),
        "Calm": ("శాంతి", "ప్రశాంతం", "సేదతీరు", "నెమ్మది", "shanti", "prashantam", "nemmadi"),
        "Energetic": ("ఉత్సాహం", "నాట్యం", "హుషారు", "జోష్", "utsaham", "natyam", "josh"),
        "Motivated": ("ధైర్యం", "విజయం", "గెలుపు", "కలలు", "ధైర్యంగా", "dhairyam", "vijayam", "gelupu", "kalalu"),
        "Nostalgic": ("జ్ఞాపకాలు", "గతం", "పాత రోజులు", "బాల్యం", "gnapakalu", "gatham", "paatha rojulu", "balyam"),
    },
    "Tamil": {
        "Sad": ("துக்கம்", "வலி", "கண்ணீர்", "தனிமை", "பிரிவு", "ஏக்கம்", "thukkam", "vali", "kanneer", "thanimai", "pirivu"),
        "Happy": ("மகிழ்ச்சி", "சிரிப்பு", "புன்னகை", "கொண்டாட்டம்", "magizhchi", "sirippu", "punnagai"),
        "Romantic": ("காதல்", "அன்பு", "காதலி", "காதலன்", "kaadhal", "kadhal", "anbu", "kaadhali"),
        "Calm": ("அமைதி", "நிம்மதி", "ஓய்வு", "மென்மை", "amaithi", "nimmathi", "oyvu"),
        "Energetic": ("ஆட்டம்", "நடனம்", "உற்சாகம்", "வேகம்", "aattam", "nadanam", "urchagam"),
        "Motivated": ("வெற்றி", "துணிவு", "கனவு", "எழுந்து", "vetri", "thunivu", "kanavu", "ezhundhu"),
        "Nostalgic": ("நினைவுகள்", "பழைய நாட்கள்", "குழந்தைப் பருவம்", "நேற்று", "ninaivugal", "pazhaiya naatkal", "netru"),
    },
    "Malayalam": {
        "Sad": ("ദുഃഖം", "വേദന", "കണ്ണീർ", "ഏകാന്തം", "വിരഹം", "duhkham", "vedana", "kanneer", "viraham"),
        "Happy": ("സന്തോഷം", "ചിരി", "പുഞ്ചിരി", "ആഘോഷം", "santhosham", "chiri", "punchiri"),
        "Romantic": ("പ്രണയം", "സ്നേഹം", "പ്രിയേ", "ഹൃദയം", "pranayam", "sneham", "priye", "hridayam"),
        "Calm": ("ശാന്തം", "സമാധാനം", "വിശ്രമം", "ശാന്തി", "shanthan", "samadhanam", "vishramam"),
        "Energetic": ("നൃത്തം", "ആവേശം", "ഉത്സാഹം", "വേഗം", "nritham", "avesham", "uthsaham"),
        "Motivated": ("വിജയം", "ധൈര്യം", "സ്വപ്നം", "പോരാട്ടം", "vijayam", "dhairyam", "swapnam", "porattam"),
        "Nostalgic": ("ഓർമ്മകൾ", "പഴയ കാലം", "ബാല്യം", "ഓർമ്മ", "ormakal", "pazhaya kaalam", "balyam"),
    },
    "Kannada": {
        "Sad": ("ದುಃಖ", "ನೋವು", "ಕಣ್ಣೀರು", "ಒಂಟಿತನ", "ವಿರಹ", "duhkha", "novu", "kanniru", "viraha"),
        "Happy": ("ಸಂತೋಷ", "ನಗು", "ಸ್ಮೈಲ್", "ಆಚರಣೆ", "santosha", "nagu", "aacharane"),
        "Romantic": ("ಪ್ರೀತಿ", "ಪ್ರೀಮ", "ಅನುರಾಗ", "ಪ್ರಿಯ", "preethi", "prema", "anuraga", "priya"),
        "Calm": ("ಶಾಂತಿ", "ನೆಮ್ಮದಿ", "ವಿಶ್ರಾಂತಿ", "ಪ್ರಶಾಂತ", "shanti", "nemmadi", "vishranthi"),
        "Energetic": ("ನೃತ್ಯ", "ಉತ್ಸಾಹ", "ಜೋಶ್", "ವೇಗ", "nruthya", "utsaha", "josh"),
        "Motivated": ("ಗೆಲುವು", "ಧೈರ್ಯ", "ಕನಸು", "ವಿಜಯ", "geluvu", "dhairya", "kanasu", "vijaya"),
        "Nostalgic": ("ನೆನಪುಗಳು", "ಹಳೆಯ ದಿನಗಳು", "ಬಾಲ್ಯ", "ನೆನಪು", "nenapugalu", "haleya dinagalu", "balya"),
    },
}


def _normalise_lyrics(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\[[0-9:.]+\]", " ", normalized)
    return " ".join(normalized.split())


def classify_lyrics(text: str | None, language: str, *, min_distinct_cues: int = 2) -> dict[str, Any]:
    """Return supported emotion labels from lyrics only; titles are never read."""
    if not isinstance(text, str) or not text.strip() or language not in _CUES:
        return {"labels": [], "confidence": {}, "evidence": {}, "status": "unknown",
                "model_version": EMOTION_MODEL_VERSION}
    normalized = _normalise_lyrics(text)
    evidence: dict[str, list[str]] = {}
    confidence: dict[str, float] = {}
    for label, cues in _CUES[language].items():
        hits = []
        seen_cues: set[str] = set()
        for cue in cues:
            cue_norm = _normalise_lyrics(cue)
            if cue_norm not in seen_cues and re.search(
                    rf"(?<![\w]){re.escape(cue_norm)}(?![\w])", normalized, flags=re.UNICODE):
                hits.append(cue)
                seen_cues.add(cue_norm)
        if len(hits) >= min_distinct_cues:
            evidence[label] = hits[:6]
            # This is an uncalibrated lexical support score, not model probability.
            confidence[label] = round(min(0.85, 0.55 + 0.08 * (len(hits) - min_distinct_cues)), 2)
    return {"labels": list(evidence), "confidence": confidence, "evidence": evidence,
            "status": "classified" if evidence else "unknown", "model_version": EMOTION_MODEL_VERSION,
            "confidence_kind": "uncalibrated_lexicon_support"}


class LocalXLMREmotionClassifier:
    """Optional offline adapter for an already fine-tuned local XLM-R checkpoint."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.environ.get("TUNELY_XLMR_EMOTION_MODEL")
        self._pipeline = None

    @property
    def available(self) -> bool:
        return bool(self.model_path and os.path.isdir(self.model_path))

    def classify(self, text: str, language: str) -> dict[str, Any]:
        if not self.available:
            return classify_lyrics(text, language)
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("text-classification", model=self.model_path,
                                      tokenizer=self.model_path, top_k=None,
                                      truncation=True, local_files_only=True)
        outputs = self._pipeline(text[:12000])
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        confidence = {str(row.get("label", "")).title(): float(row.get("score", 0))
                      for row in outputs if str(row.get("label", "")).title() in EMOTION_LABELS}
        confidence = {label: score for label, score in confidence.items() if score >= 0.65}
        return {"labels": list(confidence), "confidence": confidence,
                "evidence": {label: ["local fine-tuned XLM-RoBERTa prediction"] for label in confidence},
                "status": "classified" if confidence else "unknown",
                "model_version": f"xlmr-local:{os.path.basename(self.model_path)}",
                "confidence_kind": "model_score"}
