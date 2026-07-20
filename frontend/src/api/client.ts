import type { CompareResponse, ExplainResponse, UploadResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function parseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((item) =>
          typeof item === "object" && item && "msg" in item
            ? String((item as { msg: string }).msg)
            : JSON.stringify(item),
        )
        .join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText || "Request failed";
  }
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<UploadResponse>;
}

export async function explainImage(options: {
  file?: File;
  uploadId?: string;
  captions?: string[];
  forceBlip?: boolean;
  topN?: number;
}): Promise<ExplainResponse> {
  const form = new FormData();
  if (options.file) form.append("file", options.file);
  if (options.uploadId) form.append("upload_id", options.uploadId);
  if (options.captions?.length) {
    form.append("captions", JSON.stringify(options.captions));
  }
  if (options.forceBlip) {
    form.append("force_blip", "true");
  }
  form.append("top_n", String(options.topN ?? 8));

  const response = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<ExplainResponse>;
}

export async function fetchComparison(): Promise<CompareResponse> {
  const response = await fetch(`${API_BASE}/compare`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<CompareResponse>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
