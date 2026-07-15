export interface UploadResponse {
  filename: string;
  rows: number;
  columns: number;
}

export interface PreviewResponse {
  data: Record<string, unknown>[];
  columns: string[];
  total_rows: number;
  total_columns: number;
  preview_rows: number;
}

export interface AssignHeaderResponse {
  columns: string[];
  rows: number;
}

export interface MissingSummaryItem {
  column_name: string;
  missing_count: number;
}

export interface DuplicatesSummaryResponse {
  duplicate_count: number;
}

export interface CleanResponse {
  message: string;
  rows: number;
  columns: number;
  column_list: string[];
}

export interface ProfileReportResponse {
  report_url: string;
}

export interface CorrelationMatrixResponse {
  img_base64: string;
}

export interface TrainMetricsRegression {
  mse: number;
  rmse: number;
  mae: number;
  r2: number;
}

export interface TrainMetricsClassification {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface TrainResponse {
  model_name: string;
  task_type: "regression" | "classification";
  metrics: TrainMetricsRegression | TrainMetricsClassification;
  dropped_rows: number;
  viz_base64?: string;
  tree_base64?: string;
}

export interface LinearRegressionPayload {
  target_column: string;
  feature_columns: string[];
  test_size: number;
  random_state: number;
  fit_intercept: boolean;
}

export interface DecisionTreePayload {
  target_column: string;
  feature_columns: string[];
  test_size: number;
  random_state: number;
  max_depth: number | null;
  min_samples_split: number;
  criterion: string;
}

export interface KNNPayload {
  target_column: string;
  feature_columns: string[];
  test_size: number;
  random_state: number;
  n_neighbors: number;
  weights: string;
  metric: string;
}

export type TabId = "upload" | "clean" | "analyze" | "model" | "export";
