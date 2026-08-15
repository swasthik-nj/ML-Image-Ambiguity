import { useMemo, useState, type FormEvent, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { explainImage, uploadImage } from "../api/client";
import type { ExplainResponse } from "../types";

const LABEL_COLORS: Record<string, string> = {
  Low: "#1f8a70",
  Medium: "#e0a45c",
  High: "#c45c26",
};

export function PredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [captionText, setCaptionText] = useState(
    "a person standing in a kitchen\nsomeone cooking near a stove",
  );
  const [useBlip, setUseBlip] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainResponse | null>(null);

  const parsedCaptions = useMemo(
    () =>
      captionText
        .split(/\n|;/)
        .map((line) => line.trim())
        .filter(Boolean),
    [captionText],
  );

  const isCoco = file ? /(?:^|[/\\])0*\d{1,12}\.(?:jpe?g|png|bmp|webp)$/i.test(file.name) : false;

  function onFileChange(next: File | null) {
    setFile(next);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next ? URL.createObjectURL(next) : null);
    // COCO val filenames (e.g. 000000538236.jpg) → use human captions by default.
    if (next && /(?:^|[/\\])0*\d{1,12}\.(?:jpe?g|png|bmp|webp)$/i.test(next.name)) {
      setUseBlip(false);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Choose an image to upload.");
      return;
    }
    if (isCoco && !useBlip && parsedCaptions.length < 2) {
      setError("Enter at least two captions, or enable BLIP generation.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const uploaded = await uploadImage(file);
      const cocoCaptions = uploaded.coco_captions ?? [];
      let captionsForExplain = isCoco ? parsedCaptions : undefined;
      const forceBlip = isCoco ? useBlip : false;

      if (isCoco && !forceBlip && cocoCaptions.length >= 2) {
        captionsForExplain = cocoCaptions;
        setCaptionText(cocoCaptions.join("\n"));
      }

      const explained = await explainImage({
        uploadId: uploaded.upload_id,
        captions: captionsForExplain,
        forceBlip,
        topN: 8,
      });
      setResult(explained);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const probabilityData = result
    ? Object.entries(result.probabilities).map(([name, value]) => ({
        name,
        value: Number((value * 100).toFixed(1)),
      }))
    : [];

  const shapData =
    result?.shap_explanation.contributions.map((row) => ({
      feature: row.feature,
      shap: Number(row.shap_value.toFixed(4)),
      fill: row.shap_value >= 0 ? "#1f8a70" : "#c45c26",
    })) ?? [];

  const opencvData = result
    ? Object.entries(result.opencv_features).map(([name, value]) => ({
        name: name.replaceAll("_", " "),
        value: Number(value.toFixed(3)),
      }))
    : [];

  return (
    <div className="animate-fade space-y-8">
      <header>
        <p className="font-display text-4xl font-extrabold tracking-tight text-ink">
          Prediction
        </p>
        <p className="mt-2 max-w-2xl text-ink/65">
          Upload an image, supply captions (or let BLIP generate them), then
          view ambiguity, diversity, and SHAP charts.
        </p>
      </header>

      <form
        onSubmit={onSubmit}
        className="grid gap-6 rounded-md border border-ink/10 bg-white/70 p-5 md:grid-cols-[280px_1fr]"
      >
        <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-md border border-dashed border-ink/25 bg-fog/60 px-4 py-8 text-center transition hover:border-sea">
          {preview ? (
            <img
              src={preview}
              alt="Upload preview"
              className="max-h-56 w-full rounded-sm object-cover"
            />
          ) : (
            <span className="text-sm text-ink/55">
              Drop or click to choose an image
            </span>
          )}
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          />
          <span className="text-xs text-ink/45">
            {file ? file.name : "JPG, PNG, WEBP"}
          </span>
        </label>

        <div className="space-y-4">
          {isCoco && (
            <>
              <label className="flex items-center gap-2 text-sm text-ink/80">
                <input
                  type="checkbox"
                  checked={useBlip}
                  onChange={(event) => setUseBlip(event.target.checked)}
                />
                Generate captions with BLIP (slower on CPU)
              </label>

              {!useBlip && (
                <div>
                  <label className="mb-1 block text-sm font-medium text-ink/70">
                    Captions (one per line, min 2)
                  </label>
                  <textarea
                    value={captionText}
                    onChange={(event) => setCaptionText(event.target.value)}
                    rows={5}
                    className="w-full rounded-md border border-ink/15 bg-white px-3 py-2 text-sm text-ink outline-none ring-sea focus:ring-2"
                  />
                </div>
              )}
            </>
          )}

          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-ink px-5 py-2.5 text-sm font-semibold text-fog transition hover:bg-ink-soft disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Running explain pipeline…" : "Predict & explain"}
          </button>

          {error && (
            <p className="rounded-md border border-clay/30 bg-clay/10 px-3 py-2 text-sm text-clay">
              {error}
            </p>
          )}
        </div>
      </form>

      {result && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3">
            <Stat
              label="Predicted ambiguity"
              value={result.predicted_ambiguity}
              accent={LABEL_COLORS[result.predicted_ambiguity] ?? "#0b1f1a"}
            />
            <Stat
              label="Confidence"
              value={`${(result.confidence * 100).toFixed(1)}%`}
            />
            {result.caption_diversity && (
              <Stat
                label="Caption diversity"
                value={result.caption_diversity.caption_diversity.toFixed(3)}
              />
            )}
          </div>

          {result.captions && result.captions.length > 0 && (
            <section className="rounded-md border border-ink/10 bg-white/70 p-4">
              <h2 className="font-display text-xl font-bold text-ink">
                Generated captions
              </h2>
              {result.caption_source && (
                <p className="mt-1 text-xs text-ink/50">
                  Source:{" "}
                  {result.caption_source === "coco_human"
                    ? "COCO human captions"
                    : result.caption_source === "blip"
                      ? "BLIP (AI)"
                      : "user-provided"}
                </p>
              )}
              <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-ink/75">
                {result.captions.map((caption) => (
                  <li key={caption}>{caption}</li>
                ))}
              </ol>
            </section>
          )}

          <div className="grid gap-4 lg:grid-cols-2">
            <ChartPanel title="Class probabilities (%)">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={probabilityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#c5dcd0" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {probabilityData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={LABEL_COLORS[entry.name] ?? "#1f8a70"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel title="SHAP feature contributions">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shapData} layout="vertical" margin={{ left: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#c5dcd0" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
                    {shapData.map((entry) => (
                      <Cell key={entry.feature} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>

          <ChartPanel title="OpenCV features" tall>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={opencvData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#c5dcd0" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#2bb38a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>

          <section className="rounded-md border border-ink/10 bg-ink px-4 py-4 text-sm text-fog/90">
            <h2 className="font-display text-lg font-bold text-fog">
              SHAP summary
            </h2>
            <pre className="mt-2 whitespace-pre-wrap font-sans text-fog/80">
              {result.summary}
            </pre>
          </section>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-md border border-ink/10 bg-white/70 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-ink/50">
        {label}
      </p>
      <p
        className="mt-1 font-display text-3xl font-bold"
        style={{ color: accent ?? "#0b1f1a" }}
      >
        {value}
      </p>
    </div>
  );
}

function ChartPanel({
  title,
  children,
  tall,
}: {
  title: string;
  children: ReactNode;
  tall?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-md border border-ink/10 bg-white/70 p-4",
        tall ? "h-80" : "h-72",
      ].join(" ")}
    >
      <p className="mb-2 text-sm font-semibold text-ink">{title}</p>
      <div className="h-[85%] w-full">{children}</div>
    </div>
  );
}
