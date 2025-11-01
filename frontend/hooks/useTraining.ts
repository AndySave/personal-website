import { useState, useEffect } from "react";
import { DatasetMetadata } from "@/types/nn-framework";
import { Network } from "@/components/NetworkDropDown";

export default function useTraining(
  selectedDataset: DatasetMetadata | null,
  selectedNetwork: Network,
  userInputs: Record<string, string>
) {
  const [isTraining, setIsTraining] = useState(false);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [predictionProb, setPredictionProb] = useState<number | null>(null);

  const handleTrain = async () => {
    if (!selectedDataset) {
      return;
    }

    setIsTraining(true);
    setAccuracy(null);

    try {
      const response = await fetch(
        "http://localhost:8000/api/nn-framework/train",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify({
            dataset_name: selectedDataset.name,
            layers: selectedNetwork.layers.map((layer) => ({
              type: layer.type,
              in_features: layer.inFeatures,
              out_features: layer.outFeatures,
            })),
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Training result:", data);
      setAccuracy(data.accuracy);
    } catch (error) {
      console.error("Training failed:", error);
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async () => {
    console.log(userInputs);
    try {
      const response = await fetch(
        "http://localhost:8000/api/nn-framework/predict/adult_income",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify(userInputs),
        }
      );

      if (!response.ok) {
        throw new Error(`Server response: ${response.status}`);
      }

      const data = await response.json();
      setPredictionProb(data.probability);
      console.log(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  };

  return {
    isTraining,
    accuracy,
    predictionProb,
    handleTrain,
    handlePredict,
  };
}
