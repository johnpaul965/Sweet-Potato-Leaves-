DISEASE_INFO = {
    "Healthy": {
        "description": (
            "The leaf shows no visible signs of disease or stress. "
            "Color, texture, and structure are consistent with a healthy sweet potato plant."
        ),
        "symptoms": [
            "Deep green, uniform leaf color",
            "No spots, lesions, or discoloration",
            "Firm, intact leaf surface",
            "Normal leaf shape without curling",
        ],
        "severity": "None",
        "recommendations": [
            {
                "category": "Cultural Practices",
                "actions": [
                    "Continue regular field monitoring every 7–10 days.",
                    "Maintain appropriate plant spacing (30–40 cm between plants) for adequate air circulation.",
                    "Apply balanced NPK fertilization based on soil test results.",
                    "Ensure consistent irrigation — avoid waterlogging and drought stress.",
                ],
            },
            {
                "category": "Preventive Measures",
                "actions": [
                    "Use certified disease-free planting materials.",
                    "Rotate crops every season to reduce pathogen build-up.",
                    "Remove and destroy volunteer sweet potato plants.",
                    "Control weeds that may harbor insect vectors.",
                ],
            },
            {
                "category": "Monitoring",
                "actions": [
                    "Scout for early signs of aphids, whiteflies, and other pests.",
                    "Record plant growth and yield data for season tracking.",
                ],
            },
        ],
        "color": "#27AE60",
        "icon": "✅",
    },

    "Sweet Potato Leaf Curl Virus": {
        "description": (
            "A viral disease transmitted primarily by whiteflies (Bemisia tabaci). "
            "Affected leaves exhibit upward or downward curling, mosaic patterns, vein yellowing, "
            "and stunted plant growth. Infected plants rarely recover."
        ),
        "symptoms": [
            "Upward or downward curling of leaf margins",
            "Mosaic or mottled yellow-green discoloration",
            "Vein clearing or yellowing along leaf veins",
            "Stunted shoot and overall plant growth",
            "Distorted or crinkled leaf surface",
        ],
        "severity": "High",
        "recommendations": [
            {
                "category": "Immediate Action",
                "actions": [
                    "Remove and destroy infected plants immediately — do not compost.",
                    "Place infected material in sealed bags before disposal to prevent vector spread.",
                    "Quarantine affected field sections and restrict movement of plant material.",
                ],
            },
            {
                "category": "Vector Control (Whitefly Management)",
                "actions": [
                    "Apply imidacloprid (Confidor 200 SL) at 0.5 mL/L water via foliar spray.",
                    "Alternatively, use thiamethoxam (Actara 25 WG) at 0.2 g/L water.",
                    "Install yellow sticky traps (25/hectare) to monitor and capture adult whiteflies.",
                    "Spray insecticides in the early morning or late afternoon to protect beneficial insects.",
                    "Repeat insecticide applications every 7 days until whitefly populations are controlled.",
                ],
            },
            {
                "category": "Cultural Control",
                "actions": [
                    "Use virus-free certified planting materials only.",
                    "Plant resistant or tolerant varieties if available in your region.",
                    "Maintain a virus-free nursery with physical barriers (fine mesh nets).",
                    "Avoid planting near other Ipomoea species that may harbor the virus.",
                ],
            },
            {
                "category": "Field Sanitation",
                "actions": [
                    "Remove crop debris after harvest and plow the soil.",
                    "Observe a 2-week fallow period before replanting.",
                    "Report outbreaks to the local Bureau of Plant Industry (BPI) office.",
                ],
            },
        ],
        "color": "#E67E22",
        "icon": "⚠️",
    },

    "Fusarium Wilt": {
        "description": (
            "A soilborne fungal disease caused by Fusarium oxysporum f. sp. batatas. "
            "The pathogen invades the vascular system, blocking water transport and causing "
            "progressive wilting, yellowing, and necrosis. It persists in soil for many years."
        ),
        "symptoms": [
            "Yellowing and wilting of lower leaves progressing upward",
            "Brown or tan necrotic lesions along leaf margins and veins",
            "Dark brown discoloration of internal stem tissue when cut",
            "Premature leaf drop and stem dieback",
            "Stunted root and tuber development",
        ],
        "severity": "High",
        "recommendations": [
            {
                "category": "Fungicide Treatment",
                "actions": [
                    "Apply carbendazim (Bavistin 50 WP) at 1.0 g/L water as a soil drench.",
                    "Alternatively, use thiophanate-methyl (Topsin M) at 1.0–1.5 g/L water.",
                    "Treat planting materials with mancozeb (Dithane M-45) at 2.5 g/L water for 30 minutes before planting.",
                    "Repeat soil drenching at 14-day intervals for severe infections.",
                ],
            },
            {
                "category": "Soil Management",
                "actions": [
                    "Remove and burn infected plant material; avoid incorporating it into the soil.",
                    "Apply agricultural lime to raise soil pH to 6.5–7.0, reducing fungal activity.",
                    "Improve drainage to prevent waterlogging, which favors Fusarium survival.",
                    "Incorporate organic matter (compost) to boost beneficial soil microorganisms.",
                ],
            },
            {
                "category": "Biological Control",
                "actions": [
                    "Apply Trichoderma harzianum-based biocontrol agents to the soil at planting.",
                    "Use Bacillus subtilis-based products as a soil drench to suppress pathogen populations.",
                ],
            },
            {
                "category": "Cultural Practices",
                "actions": [
                    "Implement a minimum 3-year crop rotation with non-host crops (corn, legumes).",
                    "Plant disease-resistant varieties certified by local agricultural authorities.",
                    "Avoid moving soil and equipment from infected fields to healthy ones.",
                    "Solarize heavily infested soil by covering with transparent polyethylene for 4–6 weeks.",
                ],
            },
        ],
        "color": "#C0392B",
        "icon": "🔴",
    },

    "Cercospora Leaf Spot": {
        "description": (
            "A fungal disease caused by Cercospora bataticola or related species. "
            "Characterized by circular to irregular spots with distinct margins. "
            "Favored by warm, humid conditions and dense canopies. Can cause significant "
            "defoliation under severe infection."
        ),
        "symptoms": [
            "Small circular to irregular spots (2–10 mm diameter)",
            "Spots with tan, gray, or brown centers and dark brown to purple margins",
            "Pale yellow halo surrounding individual lesions",
            "Coalescence of multiple spots forming large necrotic patches",
            "Premature yellowing and defoliation of heavily infected leaves",
        ],
        "severity": "Moderate",
        "recommendations": [
            {
                "category": "Fungicide Application",
                "actions": [
                    "Apply mancozeb (Dithane M-45) at 2.0–2.5 g/L water as a foliar spray.",
                    "Alternatively, use chlorothalonil (Bravo 720) at 2.0 mL/L water.",
                    "For severe cases, apply azoxystrobin (Amistar 250 SC) at 0.5 mL/L water.",
                    "Begin spraying at first sign of infection; repeat every 7–10 days.",
                    "Alternate between fungicide groups to prevent resistance development.",
                ],
            },
            {
                "category": "Cultural Practices",
                "actions": [
                    "Remove and destroy heavily infected leaves from the field immediately.",
                    "Improve air circulation by appropriate plant spacing and canopy management.",
                    "Avoid overhead irrigation; use drip or furrow irrigation instead.",
                    "Irrigate in the morning so foliage dries quickly during the day.",
                ],
            },
            {
                "category": "Preventive Measures",
                "actions": [
                    "Use disease-free planting materials and resistant varieties where available.",
                    "Apply preventive fungicide sprays during warm, wet weather periods.",
                    "Maintain good field sanitation — remove crop debris after harvest.",
                    "Rotate crops for at least one season with non-host species.",
                ],
            },
            {
                "category": "Monitoring",
                "actions": [
                    "Scout fields weekly during the rainy season when humidity is high.",
                    "Record disease incidence and severity per plot for tracking purposes.",
                    "Report severe outbreaks to the local Department of Agriculture extension officer.",
                ],
            },
        ],
        "color": "#8E44AD",
        "icon": "🟣",
    },
}


def get_disease_info(class_name: str) -> dict:
    return DISEASE_INFO.get(class_name, DISEASE_INFO["Healthy"])


def get_severity_badge(severity: str) -> str:
    badges = {
        "None": "🟢 No Disease Detected",
        "Moderate": "🟡 Moderate Severity",
        "High": "🔴 High Severity",
    }
    return badges.get(severity, severity)
