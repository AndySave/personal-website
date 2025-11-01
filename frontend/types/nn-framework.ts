
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
