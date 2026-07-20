export type AmbiguityLabel = "Low" | "Medium" | "High" | string;

export interface OpenCVFeatures {
  edge_density: number;
  entropy: number;
  brightness: number;
  contrast: number;
  color_variance: number;
  texture: number;
}

export interface CaptionDiversity {
  average_similarity: number;
  minimum_similarity: number;
  maximum_similarity: number;
  std_similarity: number;
  caption_diversity: number;
  n_captions: number;
  n_pairs: number;
}

export interface ShapContribution {
  feature: string;
  value: number;
  shap_value: number;
}

export interface ShapExplanation {
  row_index: number;
  predicted_class: string;
  predicted_probability: number;
  base_value: number;
  contributions: ShapContribution[];
}

export interface ExplainResponse {
  upload_id: string | null;
  predicted_ambiguity: AmbiguityLabel;
  confidence: number;
  probabilities: Record<string, number>;
  caption_diversity: CaptionDiversity;
  opencv_features: OpenCVFeatures;
  captions: string[];
  feature_vector: Record<string, number>;
  shap_explanation: ShapExplanation;
  summary: string;
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  width: number;
  height: number;
  mode: string;
  message: string;
}

export interface CompareResponse {
  human_diversity: number | null;
  ai_diversity: number | null;
  n_human: number;
  n_ai: number;
  n_compared: number;
  human_available: boolean;
  ai_available: boolean;
  message: string;
  human_histogram?: { bin: string; count: number }[];
  ai_histogram?: { bin: string; count: number }[];
}
