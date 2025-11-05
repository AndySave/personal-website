"use client";

import { useState } from "react";
import { DatasetMetadata, NetworkMetadata } from "@/types/nn-framework";

export default function useTraining(
  selectedDataset: DatasetMetadata | null,
  selectedNetwork: NetworkMetadata | null,
  userInputs: Record<string, string>
) {
  const [epochs, setEpochs] = useState("50");
  const [modelId, setModelId] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [predictionOutput, setPredictionOutput] = useState<number | null>(null);
  const [activations, setActivations] = useState<number[][] | null>(null);
  const [trainingLoss, setTrainingLoss] = useState<number[] | null>(null);
  const [trainingAccuracy, setTrainingAccuracy] = useState<number[] | null>(
    null
  );

  const API_URL = process.env.NEXT_PUBLIC_API_URL;

  const handleTrain = async () => {
    if (!selectedDataset || !selectedNetwork) {
      return;
    }

    setIsTraining(true);
    setModelId(null);
    setTrainingLoss(null);
    setTrainingAccuracy(null);

    try {
      const response = await fetch(`${API_URL}/api/nn-framework/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          epochs: epochs,
          dataset_name: selectedDataset.name,
          model_size: selectedNetwork.model_size,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Training result:", data);
      setModelId(data.model_id);
      setTrainingLoss(data.training_loss);
      setTrainingAccuracy(data.training_accuracy);
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
        `${API_URL}/api/nn-framework/predict/${selectedDataset?.name}?model_id=${modelId}`,
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
      setPredictionOutput(data.output);
      setActivations(data.activations);
      console.log(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  };

  return {
    epochs,
    setEpochs,
    isTraining,
    predictionOutput,
    activations,
    trainingLoss,
    trainingAccuracy,
    handleTrain,
    handlePredict,
  };
}
