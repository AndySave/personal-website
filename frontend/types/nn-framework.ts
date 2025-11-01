
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
  description: string;
  features: FeatureMetadata[];
}

export type Layer = {
  type: string;
  inFeatures?: number;
  outFeatures?: number;
}

export type Network = {
  id: number;
  name: string;
  layers: Layer[];
}
