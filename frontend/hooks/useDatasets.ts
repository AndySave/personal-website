import { DatasetMetadata } from "@/types/nn-framework";
import { useState, useEffect } from "react";

const getDefaultInputs = (dataset: DatasetMetadata): Record<string, string> =>
  Object.fromEntries(
    dataset.features.map((feature) => [
      feature.name,
      feature.options?.[0].value ?? "",
    ])
  );

export default function useDatasets() {
  const [userInputs, setUserInputs] = useState<Record<string, string>>({});
  const [selectedDataset, setSelectedDataset] =
    useState<DatasetMetadata | null>(null);
  const [datasets, setDatasets] = useState<DatasetMetadata[] | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/nn-framework/datasets-metadata")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Server error ${res.status}`);
        }
        return res.json();
      })
      .then((data: DatasetMetadata[]) => {
        setDatasets(data);
        setSelectedDataset(data[0]);
        setUserInputs(getDefaultInputs(data[0]));
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {  // TODO: Temp for debug
    if (datasets) {
      console.log("Datasets updated:", datasets);
    }
  }, [datasets]);

  return {
    datasets,
    selectedDataset,
    setSelectedDataset,
    userInputs,
    setUserInputs,
  };
}
