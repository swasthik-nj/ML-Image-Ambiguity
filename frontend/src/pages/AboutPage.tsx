const stack = [
  { name: "FastAPI", role: "REST backend for upload, predict, explain" },
  { name: "BLIP", role: "AI caption generation when captions are omitted" },
  { name: "Sentence-BERT", role: "Caption embeddings for diversity" },
  { name: "OpenCV", role: "Edge, entropy, brightness, contrast, texture" },
  { name: "XGBoost / RF", role: "Ambiguity classification" },
  { name: "SHAP", role: "Feature attribution for each prediction" },
  { name: "React + Tailwind", role: "Responsive research demo UI" },
  { name: "Recharts", role: "Probability, SHAP, and comparison charts" },
];

export function AboutPage() {
  return (
    <div className="animate-fade mx-auto max-w-3xl space-y-8">
      <header>
        <p className="font-display text-4xl font-extrabold tracking-tight text-ink">
          About
        </p>
        <p className="mt-3 text-lg text-ink/70">
          This demo is part of an MCA research project on explainable image
          ambiguity prediction using human and AI-generated caption diversity
          with computer vision features.
        </p>
      </header>

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-bold text-ink">Idea</h2>
        <p className="leading-relaxed text-ink/70">
          When captions for the same image disagree, the scene is often
          ambiguous. We turn that disagreement into a numeric diversity score,
          combine it with classical OpenCV signals, predict Low / Medium / High
          ambiguity, and explain the decision with SHAP.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-ink">Stack</h2>
        <ul className="divide-y divide-ink/10 border-y border-ink/10">
          {stack.map((item) => (
            <li
              key={item.name}
              className="flex flex-col gap-1 py-3 sm:flex-row sm:items-baseline sm:justify-between"
            >
              <span className="font-semibold text-ink">{item.name}</span>
              <span className="text-sm text-ink/60 sm:text-right">
                {item.role}
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-bold text-ink">Run locally</h2>
        <pre className="overflow-x-auto rounded-md bg-ink px-4 py-3 text-sm text-fog">
{`# API
$env:PYTHONPATH="src;."
uvicorn app:app --reload

# UI
cd frontend
npm install
npm run dev`}
        </pre>
      </section>
    </div>
  );
}
