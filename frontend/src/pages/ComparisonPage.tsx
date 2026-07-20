import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchComparison } from "../api/client";
import type { CompareResponse } from "../types";

export function ComparisonPage() {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await fetchComparison();
        if (!cancelled) setData(result);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load comparison");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const meanChart = [
    {
      name: "Mean diversity",
      Human: data?.human_diversity ?? 0,
      AI: data?.ai_diversity ?? 0,
    },
  ];

  return (
    <div className="animate-fade space-y-8">
      <header>
        <p className="font-display text-4xl font-extrabold tracking-tight text-ink">
          Comparison
        </p>
        <p className="mt-2 max-w-2xl text-ink/65">
          Human COCO caption diversity versus BLIP AI caption diversity across
          the shared dataset sample.
        </p>
      </header>

      {loading && <p className="text-ink/60">Loading comparison metrics…</p>}
      {error && (
        <p className="rounded-md border border-clay/30 bg-clay/10 px-4 py-3 text-sm text-clay">
          {error}. Start the API with <code>uvicorn app:app --reload</code>.
        </p>
      )}

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <MetricTile
              label="Human Diversity"
              value={data.human_diversity}
              sub={`${data.n_human} images`}
              available={data.human_available}
            />
            <MetricTile
              label="AI Diversity"
              value={data.ai_diversity}
              sub={`${data.n_ai} images`}
              available={data.ai_available}
            />
          </div>

          <p className="text-sm text-ink/55">{data.message}</p>

          <div className="h-80 w-full rounded-md border border-ink/10 bg-white/60 p-4">
            <p className="mb-3 text-sm font-semibold text-ink">
              Mean caption diversity
            </p>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={meanChart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#c5dcd0" />
                <XAxis dataKey="name" tick={{ fill: "#0b1f1a" }} />
                <YAxis domain={[0, 1]} tick={{ fill: "#0b1f1a" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Human" fill="#1f8a70" radius={[4, 4, 0, 0]} />
                <Bar dataKey="AI" fill="#e0a45c" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {(data.human_histogram?.length || data.ai_histogram?.length) && (
            <div className="grid gap-4 lg:grid-cols-2">
              <HistogramPanel
                title="Human diversity distribution"
                color="#1f8a70"
                rows={data.human_histogram ?? []}
              />
              <HistogramPanel
                title="AI diversity distribution"
                color="#e0a45c"
                rows={data.ai_histogram ?? []}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MetricTile({
  label,
  value,
  sub,
  available,
}: {
  label: string;
  value: number | null;
  sub: string;
  available: boolean;
}) {
  return (
    <div className="rounded-md border border-ink/10 bg-white/70 px-5 py-4">
      <p className="text-sm font-medium text-ink/55">{label}</p>
      <p className="mt-1 font-display text-4xl font-bold text-ink">
        {available && value != null ? value.toFixed(2) : "—"}
      </p>
      <p className="mt-1 text-xs text-ink/45">{sub}</p>
    </div>
  );
}

function HistogramPanel({
  title,
  color,
  rows,
}: {
  title: string;
  color: string;
  rows: { bin: string; count: number }[];
}) {
  return (
    <div className="h-72 rounded-md border border-ink/10 bg-white/60 p-4">
      <p className="mb-2 text-sm font-semibold text-ink">{title}</p>
      {rows.length === 0 ? (
        <p className="text-sm text-ink/50">Dataset not available.</p>
      ) : (
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#c5dcd0" />
            <XAxis dataKey="bin" tick={{ fill: "#0b1f1a", fontSize: 11 }} />
            <YAxis allowDecimals={false} tick={{ fill: "#0b1f1a" }} />
            <Tooltip />
            <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
