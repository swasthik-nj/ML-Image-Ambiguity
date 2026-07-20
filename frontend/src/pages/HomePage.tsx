import { Link } from "react-router-dom";

export function HomePage() {
  return (
    <section className="relative overflow-hidden">
      <div className="pointer-events-none absolute -right-24 top-8 h-72 w-72 rounded-full bg-sea-bright/20 blur-3xl" />
      <div className="pointer-events-none absolute -left-16 bottom-0 h-56 w-56 rounded-full bg-amber/25 blur-3xl" />

      <div className="relative grid min-h-[70vh] items-center gap-10 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="animate-rise">
          <p className="font-display text-5xl font-extrabold leading-[0.95] tracking-tight text-ink sm:text-6xl md:text-7xl">
            Ambiguity Lens
          </p>
          <h1 className="mt-5 max-w-xl text-xl font-medium text-ink/80 sm:text-2xl">
            See how captions disagree — and why a model calls an image Low,
            Medium, or High ambiguity.
          </h1>
          <p className="mt-4 max-w-lg text-base leading-relaxed text-ink/60">
            Upload a photo, generate captions, inspect diversity and OpenCV
            signals, then read a SHAP explanation of the prediction.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              to="/prediction"
              className="rounded-md bg-ink px-5 py-2.5 text-sm font-semibold text-fog transition hover:bg-ink-soft"
            >
              Run prediction
            </Link>
            <Link
              to="/comparison"
              className="rounded-md border border-ink/20 bg-white/50 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-ink/40 hover:bg-white/80"
            >
              Compare human vs AI
            </Link>
          </div>
        </div>

        <div className="animate-rise-delay relative">
          <div
            className="aspect-[4/5] w-full max-w-md overflow-hidden rounded-sm bg-ink shadow-xl shadow-ink/20"
            style={{
              backgroundImage:
                "linear-gradient(145deg, #143029 0%, #1f8a70 48%, #e0a45c 140%)",
            }}
          >
            <div className="flex h-full flex-col justify-between p-6 text-fog">
              <p className="font-display text-sm uppercase tracking-[0.2em] text-fog/70">
                Live research UI
              </p>
              <div>
                <p className="font-display text-4xl font-bold leading-none">
                  Caption
                  <br />
                  disagreement
                  <br />
                  made visible
                </p>
                <p className="mt-4 max-w-xs text-sm text-fog/75">
                  Human COCO captions · BLIP AI captions · OpenCV · SHAP
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
