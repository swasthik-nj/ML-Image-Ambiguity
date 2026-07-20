# Ambiguity Lens (React frontend)

Modern React UI for the Image Ambiguity Prediction API.

## Pages

- **Home** — brand landing
- **Prediction** — upload image, captions / BLIP, ambiguity + SHAP charts
- **Comparison** — human vs AI caption diversity charts
- **About** — project stack and run instructions

## Stack

- React + TypeScript + Vite
- Tailwind CSS v4
- Recharts
- React Router

## Run

From the `frontend/` folder:

```bash
npm install
npm run dev
```

Open http://localhost:5173

Start the API in another terminal (project root):

```powershell
$env:PYTHONPATH="src;."
uvicorn app:app --reload
```

Vite proxies `/api/*` → `http://127.0.0.1:8000/*`.
