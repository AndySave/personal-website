export interface FeatureOptions {
  value: string;
  display_name: string;
}

export interface FeatureMetadata {
  name: string;
  display_name: string;
  type: "categorical" | "numeric";
  options?: FeatureOptions[];
  min?: number;
  max?: number;
}

export interface DatasetMetadata {
  name: string;
  display_name: string;
  task_type: "regression" | "binary_classification" | "multi_classification";
  description: string;
  features: FeatureMetadata[];
}

export type Layer = {
  type: string;
  inFeatures?: number;
  outFeatures?: number;
};

export type Network = {
  id: number;
  name: string;
  model_size: "small" | "medium" | "large";
};

export interface NetworkMetadata {
  display_name: string;
  model_size: "small" | "medium" | "large";
  layer_sizes: number[];
}
