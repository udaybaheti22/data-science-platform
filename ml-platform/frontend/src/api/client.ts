import type {
  UploadResponse,
  AssignHeaderResponse,
  PreviewResponse,
  MissingSummaryItem,
  DuplicatesSummaryResponse,
  CleanResponse,
  ProfileReportResponse,
  CorrelationMatrixResponse,
  TrainResponse,
  LinearRegressionPayload,
  DecisionTreePayload,
  KNNPayload,
  AISuggestionResponse,
  AISuggestPayload,
} from "../types/api";

const BASE_URL = import.meta.env.VITE_API_URL as string;

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api = {
  async upload(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<UploadResponse>(response);
  },

  async assignHeader(): Promise<AssignHeaderResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/assign_header`, {
        method: "POST",
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<AssignHeaderResponse>(response);
  },

  async getPreview(): Promise<PreviewResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/preview`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<PreviewResponse>(response);
  },

  async getMissingSummary(): Promise<MissingSummaryItem[]> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/missing_summary`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<MissingSummaryItem[]>(response);
  },

  async getDuplicatesSummary(): Promise<DuplicatesSummaryResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/duplicates_summary`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<DuplicatesSummaryResponse>(response);
  },

  async changeType(payload: {
    column_name: string;
    new_type: string;
  }): Promise<CleanResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/change_type`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<CleanResponse>(response);
  },

  async renameColumn(payload: {
    old_column_name: string;
    new_column_name: string;
  }): Promise<CleanResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/rename_column`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<CleanResponse>(response);
  },

  async clean(payload: {
    operations: Array<{ type: string; columns?: string[]; method?: string }>;
  }): Promise<CleanResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/clean`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<CleanResponse>(response);
  },

  async getProfileReport(): Promise<ProfileReportResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/profile_report`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<ProfileReportResponse>(response);
  },

  async getCorrelationMatrix(): Promise<CorrelationMatrixResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/correlation_matrix`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<CorrelationMatrixResponse>(response);
  },

  async trainLinearRegression(
    payload: LinearRegressionPayload
  ): Promise<TrainResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/model/linear_regression`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<TrainResponse>(response);
  },

  async trainDecisionTree(payload: DecisionTreePayload): Promise<TrainResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/model/decision_tree`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<TrainResponse>(response);
  },

  async trainKNN(payload: KNNPayload): Promise<TrainResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/model/knn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<TrainResponse>(response);
  },

  async exportDataset(): Promise<Blob> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/data/export`);
    } catch {
      throw new Error("Could not connect to backend");
    }
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "Request failed");
    }
    return response.blob();
  },

  async getAISuggestions(payload: AISuggestPayload): Promise<AISuggestionResponse> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/suggest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new Error("Could not connect to backend");
    }
    return handleResponse<AISuggestionResponse>(response);
  },
};
