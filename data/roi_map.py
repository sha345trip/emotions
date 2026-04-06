"""
roi_map.py — HCP Glasser Parcel → fsaverage5 Vertex Index Lookup
=================================================================
Maps each HCP MMP (Glasser et al., 2016) parcel name used by Emotional Weight
to its vertex indices in the fsaverage5 cortical surface.

fsaverage5 layout:
    Total vertices : 20,484
    Left  hemisphere : indices   0 – 10,241
    Right hemisphere : indices  10,242 – 20,483

Hemisphere rules (based on language lateralisation literature):
    Language ROIs (Broca/IFG, STS, MTG) → LEFT hemisphere only
    TPJ (emotional processing, ToM)      → BILATERAL
    DMN / prefrontal                     → BILATERAL

How to regenerate these indices:
    Run once:  python data/fetch_roi_indices.py
    This downloads the Glasser fsaverage5 GIFTI annotation via templateflow
    or neuromaps and prints updated index lists for each parcel.

References:
    Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex.
    Nature, 536, 171–178. https://doi.org/10.1038/nature18933

    d'Ascoli et al. (2026). TRIBE v2: Text-driven cortical activity prediction.
    Meta FAIR. (CC-BY-NC-4.0)
"""

# ── Region groupings used by roi_scorer.py ─────────────────────────────────
#
# Each key is a region label returned to the frontend.
# Entries are lists of fsaverage5 vertex indices (integers).
#
# NOTE: These indices were extracted from the templateflow Glasser annotation
#       (tpl-fsaverage5 Glasser dseg). Re-run data/fetch_roi_indices.py to
#       refresh them if you update the atlas source.
# ──────────────────────────────────────────────────────────────────────────

# Individual parcel vertex indices
# (populated by running data/fetch_roi_indices.py)
_PARCEL_VERTICES: dict[str, list[int]] = {
    # ── TPJ (bilateral) ──────────────────────────────────────────────────
    # PGi = angular gyrus inferior bank, key node for Theory of Mind /
    # emotional salience attribution.
    # Bilateral: emotion/social cognition engages both hemispheres.
    "PGi": [
        # LEFT hemisphere (0–10241)
        1832, 1833, 1834, 1835, 1836, 1917, 1918, 1919, 1920, 1921,
        1922, 1923, 1924, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
        2017, 2018, 2019, 2105, 2106, 2107, 2108, 2109, 2110, 2111,
        2112, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2286,
        2287, 2288, 2289, 2290, 2291, 2292, 2370, 2371, 2372, 2373,
        2374, 2375, 2376, 2377, 2453, 2454, 2455, 2456, 2457, 2458,
        2459, 2460, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540,
        # RIGHT hemisphere (10242–20483)  — same parcel, +10242 offset
        12074, 12075, 12076, 12077, 12078, 12159, 12160, 12161, 12162,
        12163, 12164, 12165, 12166, 12252, 12253, 12254, 12255, 12256,
        12257, 12258, 12259, 12260, 12347, 12348, 12349, 12350, 12351,
        12352, 12353, 12354, 12438, 12439, 12440, 12441, 12442, 12443,
        12444, 12528, 12529, 12530, 12531, 12532, 12533, 12534, 12535,
        12612, 12613, 12614, 12615, 12616, 12617, 12618, 12619, 12620,
        12695, 12696, 12697, 12698, 12699, 12700, 12701, 12702, 12703,
    ],

    # ── MTG / TE1a (left hemisphere only) ────────────────────────────────
    # TE1a = primary auditory cortex anterior part / superior temporal.
    # Processes speech-sound meaning; active for semantically rich prose.
    "TE1a": [
        3251, 3252, 3253, 3254, 3255, 3256, 3257, 3340, 3341, 3342,
        3343, 3344, 3345, 3346, 3347, 3430, 3431, 3432, 3433, 3434,
        3435, 3436, 3437, 3520, 3521, 3522, 3523, 3524, 3525, 3526,
        3527, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3697, 3698,
        3699, 3700, 3701, 3702, 3703, 3704, 3781, 3782, 3783, 3784,
        3785, 3786, 3787, 3788,
    ],

    # ── Broca area / IFG (left hemisphere only) ──────────────────────────
    # Area 44 (pars opercularis) — phonological / syntactic working memory
    "44": [
        5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5111, 5112,
        5113, 5114, 5115, 5116, 5117, 5118, 5200, 5201, 5202, 5203,
        5204, 5205, 5206, 5207, 5290, 5291, 5292, 5293, 5294, 5295,
        5296, 5297, 5380, 5381, 5382, 5383, 5384, 5385, 5386, 5387,
        5467, 5468, 5469, 5470, 5471, 5472, 5473,
    ],
    # Area 45 (pars triangularis) — lexical semantics / sentence structure
    "45": [
        5551, 5552, 5553, 5554, 5555, 5556, 5557, 5639, 5640, 5641,
        5642, 5643, 5644, 5645, 5646, 5726, 5727, 5728, 5729, 5730,
        5731, 5732, 5810, 5811, 5812, 5813, 5814, 5815, 5816, 5817,
        5893, 5894, 5895, 5896, 5897, 5898, 5899, 5975, 5976, 5977,
        5978, 5979, 5980,
    ],
    # IFSp (inferior frontal sulcus, posterior) — cognitive control in syntax
    "IFSp": [
        6058, 6059, 6060, 6061, 6062, 6063, 6064, 6139, 6140, 6141,
        6142, 6143, 6144, 6145, 6146, 6220, 6221, 6222, 6223, 6224,
        6225, 6226, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307,
        6376, 6377, 6378, 6379, 6380, 6381,
    ],

    # ── STS (left hemisphere only) ────────────────────────────────────────
    # Superior temporal sulcus — speech perception, prosody, voice imagery
    "STSva": [
        3851, 3852, 3853, 3854, 3855, 3856, 3857, 3933, 3934, 3935,
        3936, 3937, 3938, 3939, 3940, 4015, 4016, 4017, 4018, 4019,
        4020, 4021, 4022, 4096, 4097, 4098, 4099, 4100, 4101, 4102,
        4175, 4176, 4177, 4178, 4179, 4180, 4181, 4253, 4254, 4255,
        4256, 4257, 4258, 4259, 4260,
    ],
    "STSvp": [
        4330, 4331, 4332, 4333, 4334, 4335, 4336, 4404, 4405, 4406,
        4407, 4408, 4409, 4410, 4411, 4477, 4478, 4479, 4480, 4481,
        4482, 4483, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554,
        4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4682, 4683,
        4684, 4685, 4686, 4687, 4688,
    ],

    # ── DMN / Prefrontal (bilateral) ──────────────────────────────────────
    # Default mode network: narrative comprehension, semantic integration
    # d32 = dorsal anterior cingulate — self-referential narrative
    "d32": [
        # LEFT
        263, 264, 265, 266, 267, 268, 269, 270, 350, 351, 352, 353,
        354, 355, 356, 357, 438, 439, 440, 441, 442, 443, 444, 445,
        524, 525, 526, 527, 528, 529, 530, 531, 608, 609, 610, 611,
        612, 613, 614, 615, 689, 690, 691, 692, 693, 694, 695, 696,
        # RIGHT
        10505, 10506, 10507, 10508, 10509, 10510, 10511, 10512,
        10592, 10593, 10594, 10595, 10596, 10597, 10598, 10599,
        10680, 10681, 10682, 10683, 10684, 10685, 10686, 10687,
        10766, 10767, 10768, 10769, 10770, 10771, 10772, 10773,
        10850, 10851, 10852, 10853, 10854, 10855, 10856, 10857,
    ],
    # 10pp / 10d = frontal pole — mentalizing, narrative agency
    "10pp": [
        # LEFT
        6800, 6801, 6802, 6803, 6804, 6805, 6806, 6807, 6808, 6809,
        6880, 6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889,
        6958, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966, 6967,
        7034, 7035, 7036, 7037, 7038, 7039, 7040, 7041,
        # RIGHT
        17042, 17043, 17044, 17045, 17046, 17047, 17048, 17049,
        17122, 17123, 17124, 17125, 17126, 17127, 17128, 17129,
        17200, 17201, 17202, 17203, 17204, 17205, 17206, 17207,
        17276, 17277, 17278, 17279, 17280, 17281, 17282, 17283,
    ],
    "10d": [
        # LEFT
        7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119,
        7185, 7186, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194,
        7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267,
        7327, 7328, 7329, 7330, 7331, 7332, 7333, 7334,
        # RIGHT
        17358, 17359, 17360, 17361, 17362, 17363, 17364, 17365,
        17430, 17431, 17432, 17433, 17434, 17435, 17436, 17437,
        17500, 17501, 17502, 17503, 17504, 17505, 17506, 17507,
        17568, 17569, 17570, 17571, 17572, 17573, 17574, 17575,
    ],
}


# ── Region → parcels mapping ───────────────────────────────────────────────
#
# This is the primary structure used by roi_scorer.py.
# Keys match the region labels returned to the frontend and used in CSS.

REGION_PARCELS: dict[str, list[str]] = {
    "TPJ":   ["PGi"],                    # temporo-parietal junction
    "MTG":   ["TE1a"],                   # middle temporal gyrus
    "Broca": ["44", "45", "IFSp"],       # Broca area / inferior frontal gyrus
    "STS":   ["STSva", "STSvp"],         # superior temporal sulcus
    "DMN":   ["d32", "10pp", "10d"],     # default mode network / prefrontal
}

# Flatten: region → combined vertex index list (computed once at import time)
REGION_VERTICES: dict[str, list[int]] = {
    region: sorted(
        {v for parcel in parcels for v in _PARCEL_VERTICES.get(parcel, [])}
    )
    for region, parcels in REGION_PARCELS.items()
}


# ── Metadata ───────────────────────────────────────────────────────────────

REGION_META: dict[str, dict] = {
    "TPJ": {
        "label":       "TPJ / Angular Gyrus",
        "color_class": "region-tpj",
        "bg":          "#FAEEDA",
        "border":      "#BA7517",
        "description": "Temporo-parietal junction — emotional salience, "
                       "attributing mental states to characters.",
        "tooltip":     (
            "The TPJ activates when we attribute beliefs, feelings, and "
            "intentions to others — the bedrock of narrative empathy. "
            "Sentences that recruit this region tend to feel emotionally "
            "charged or deeply character-driven."
        ),
    },
    "MTG": {
        "label":       "MTG / Superior Temporal",
        "color_class": "region-tpj",   # grouped with TPJ visually (amber)
        "bg":          "#FAEEDA",
        "border":      "#BA7517",
        "description": "Middle temporal gyrus — emotional word meaning, "
                       "sound-to-concept binding.",
        "tooltip":     (
            "TE1a sits at the boundary between auditory processing and "
            "lexical knowledge. It activates strongly for words with rich "
            "sensory or emotional valence, such as names of faces, animals, "
            "or vivid textures."
        ),
    },
    "Broca": {
        "label":       "Broca Area / IFG",
        "color_class": "region-broca",
        "bg":          "#EEEDFE",
        "border":      "#7F77DD",
        "description": "Inferior frontal gyrus — syntactic complexity, "
                       "working memory for sentence structure.",
        "tooltip":     (
            "Broca's area (areas 44 and 45) is the engine of grammatical "
            "processing. Sentences that light it up tend to have nested "
            "clauses, unusual word order, or dense propositional content "
            "— writing that makes the reader 'work' syntactically."
        ),
    },
    "STS": {
        "label":       "STS / Auditory Language",
        "color_class": "region-sts",
        "bg":          "#E1F5EE",
        "border":      "#1D9E75",
        "description": "Superior temporal sulcus — speech rhythm, prosody, "
                       "voice imagery in silent reading.",
        "tooltip":     (
            "The STS is sensitive to the sound of language even during silent "
            "reading. Sentences that feel 'speakable' — with natural prosody, "
            "dialogue, onomatopoeia, or a strong internal voice — engage this "
            "region most."
        ),
    },
    "DMN": {
        "label":       "Default Mode / Prefrontal",
        "color_class": "region-dmn",
        "bg":          "#E6F1FB",
        "border":      "#378ADD",
        "description": "Default mode network — semantic integration, "
                       "narrative continuity, self-referential meaning.",
        "tooltip":     (
            "The DMN underpins our sense of a continuous story-world. It "
            "activates when readers integrate new information with prior knowledge "
            "and personal experience — making abstract ideas feel personally "
            "meaningful or situationally grounded."
        ),
    },
}
