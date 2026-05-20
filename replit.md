# Sweet Potato Leaf Health Detection System

An image-based CNN diagnostic system that classifies sweet potato (*Ipomoea batatas*) leaf health conditions into four categories using MobileNetV2 and GradCAM explainability.

## Run & Operate

- `cd artifacts/sweet-potato-detector && streamlit run app.py` — run the Streamlit app (port 5000)
- Workflow name: **Sweet Potato Detector**

### Backend / API (pre-existing)
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- **Frontend/App:** Python 3.11, Streamlit 1.41
- **ML:** TensorFlow 2.17, MobileNetV2 (ImageNet pretrained + fine-tunable)
- **Visualization:** Plotly, Matplotlib, OpenCV
- **pnpm workspaces, Node.js 24, TypeScript 5.9** (pre-existing infrastructure)
- **API:** Express 5 | **DB:** PostgreSQL + Drizzle ORM

## Where things live

- `artifacts/sweet-potato-detector/app.py` — main Streamlit application
- `artifacts/sweet-potato-detector/model.py` — MobileNetV2 model definition, loading, and inference
- `artifacts/sweet-potato-detector/gradcam.py` — GradCAM heatmap generation and overlay
- `artifacts/sweet-potato-detector/recommendations.py` — disease-specific management knowledge base
- `artifacts/sweet-potato-detector/sweet_potato_model.h5` — *(place trained weights here)*
- `lib/api-spec/openapi.yaml` — API contract source of truth

## Architecture decisions

- **Demo/simulation mode:** When no `sweet_potato_model.h5` file is found, the system runs a color-feature-based simulation. This demonstrates the full UI pipeline without requiring trained weights.
- **GradCAM implementation:** Uses TensorFlow's `GradientTape` against the last Conv2D layer. In demo mode, a color+edge saliency proxy is used.
- **Prescriptive module:** Recommendations are a static knowledge base sourced from Philippine BPI/DA crop protection guidelines, mapped per disease class.
- **Transfer learning architecture:** MobileNetV2 (ImageNet) → GlobalAveragePooling2D → Dropout(0.3) → Dense(4, softmax). Base layers frozen; fine-tune on sweet potato dataset.
- **Four classification targets:** Healthy, Sweet Potato Leaf Curl Virus, Fusarium Wilt, Cercospora Leaf Spot.

## Product

Users upload a sweet potato leaf image and receive:
1. **Descriptive analytics** — identified health condition with confidence score and observed symptom list
2. **Diagnostic analytics** — GradCAM heatmap showing which leaf regions drove the decision
3. **Predictive analytics** — spread risk assessment for the detected condition
4. **Prescriptive analytics** — evidence-based management recommendations (fungicides, cultural control, quarantine)

## User preferences

- System follows the research paper: *Billante & Querioso, LNU 2024*
- Four disease classes per the paper's objectives

## Gotchas

- TensorFlow logs CUDA warnings on startup (no GPU) — these are harmless; the model runs on CPU.
- To enable real inference: place your fine-tuned weights at `artifacts/sweet-potato-detector/sweet_potato_model.h5`.
- Streamlit requires `--server.port 5000` via `.streamlit/config.toml` (already configured).

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
