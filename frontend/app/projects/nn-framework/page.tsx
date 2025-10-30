"use client";

import { useEffect, useState } from "react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import NetworkDropDown, { Network } from "@/components/NetworkDropDown";
import DatasetDropDown, { Dataset } from "@/components/DatasetDropDown";
import TrainButton from "./TrainButton";
import DropDownMenu from "@/components/DropDownMenu";

const NETWORKS: Network[] = [
  {
    id: 1,
    name: "Small network",
    layers: [
      { type: "dense", inFeatures: 31, outFeatures: 4 },
      { type: "relu" },
      { type: "dense", inFeatures: 4, outFeatures: 6 },
      { type: "relu" },
      { type: "dense", inFeatures: 6, outFeatures: 1 },
      { type: "sigmoid" },
    ],
  },
  {
    id: 2,
    name: "Medium network",
    layers: [
      { type: "dense", inFeatures: 31, outFeatures: 8 },
      { type: "relu" },
      { type: "dense", inFeatures: 8, outFeatures: 16 },
      { type: "relu" },
      { type: "dense", inFeatures: 16, outFeatures: 8 },
      { type: "relu" },
      { type: "dense", inFeatures: 8, outFeatures: 1 },
      { type: "sigmoid" },
    ],
  },
];

const DATASETS: Dataset[] = [
  {
    id: 1,
    name: "Income in America",
    inputs: [
      {
        name: "age",
        options: ["18", "25", "30", "40", "50", "60"],
      },
      {
        name: "workclass",
        options: [
          "Federal government",
          "State government",
          "Local government",
          "Never worked",
          "Private",
          "Self-employed (incorporated)",
          "Self-employed (not incorporated)",
          "Without pay",
        ],
      },
      {
        name: "education",
        options: [
          "Below high school",
          "High school",
          "Some college",
          "Bachelors",
          "Masters",
          "Doctorate (PhD/EdD)",
        ],
      },
      {
        name: "marital status",
        options: [
          "Never married",
          "Married (civilian spouse)",
          "Married (Armed Forces spouse)",
          "Married (spouse living elsewhere)",
          "Separated",
          "Divorced",
          "Widowed",
        ],
      },
      {
        name: "race",
        options: ["White", "Black", "Asian", "Native American", "Other"],
      },
      {
        name: "sex",
        options: ["Male", "Female"],
      },
      {
        name: "work hours",
        options: ["20", "30", "40", "50", "60"],
      },
    ],
  },
];

const incomeInAmericaMapping: Record<string, string> = {
    "Federal government": "Federal-gov",
    "State government": "State-gov",
    "Local government": "Local-gov",
    "Never worked": "Never-worked",
    "Private": "Private",
    "Self-employed (incorporated)": "Self-emp-inc",
    "Self-employed (not incorporated)": "Self-emp-not-inc",
    "Without pay": "Without-pay",

    "Below high school": "<HS",
    "High school": "HS-grad",
    "Some college": "Some-college",
    "Bachelors": "Bachelors",
    "Masters": "Masters",
    "Doctorate (PhD/EdD)": "Doctorate",

    "Never married": "Never-married",
    "Married (civilian spouse)": "Married-civ-spouse",
    "Married (Armed Forces spouse)": "Married-AF-spouse",
    "Married (spouse living elsewhere)": "Married-spouse-absent",
    "Separated": "Separated",
    "Divorced": "Divorced",
    "Widowed": "Widowed",

    "White": "White", 
    "Black": "Black", 
    "Asian": "Asian-Pac-Islander", 
    "Native American": "Amer-Indian-Eskimo",
    "Other": "Other",

    "Male": "Male",
    "Female": "Female",
}

const getLayerSizes = (network: Network, dataset: Dataset) => {
  const layerSizes = [dataset.inputs.length];

  for (const layer of network.layers) {
    if (layer.outFeatures !== undefined) {
      layerSizes.push(layer.outFeatures);
    }
  }

  return layerSizes;
};

const getDefaultInputs = (dataset: Dataset): Record<string, string> =>
    Object.fromEntries(
        dataset.inputs.map((input) => [input.name, input.options?.[0] ?? ""])
    );

export default function NeuralNetworkPage() {
  const [selectedNetwork, setSelectedNetwork] = useState<Network>(NETWORKS[0]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset>(DATASETS[0]);
  const [isTraining, setIsTraining] = useState(false);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [predictionProb, setPredictionProb] = useState<number | null>(null);
  const [userInputs, setUserInputs] = useState<Record<string, string>>(getDefaultInputs(DATASETS[0]));
  
  useEffect(() => {
    setUserInputs(getDefaultInputs(selectedDataset));
  }, [selectedDataset]);

  const handleTrain = async () => {
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
        "http://localhost:8000/api/nn-framework/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify({
            age: userInputs["age"],
            workclass: incomeInAmericaMapping[userInputs["workclass"]],
            education: incomeInAmericaMapping[userInputs["education"]],
            marital_status: incomeInAmericaMapping[userInputs["marital status"]],
            race: incomeInAmericaMapping[userInputs["race"]],
            sex: incomeInAmericaMapping[userInputs["sex"]],
            work_hours: userInputs["work hours"],
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Server response: ${response.status}`);
      }

      const data = await response.json();
      setPredictionProb(data.probability)
      console.log(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  };

  return (
    <main className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col justify-center items-center p-20">
        <div className="flex pb-5 gap-10">
          <NetworkDropDown
            options={NETWORKS}
            value={selectedNetwork}
            onChange={setSelectedNetwork}
          />
          <DatasetDropDown
            options={DATASETS}
            dataset={selectedDataset}
            onChange={setSelectedDataset}
          />
        </div>
        <NeuralNetworkVisualizer
          layerSizes={getLayerSizes(selectedNetwork, selectedDataset)}
        />
        <TrainButton onClick={handleTrain}>Train!</TrainButton>
        <TrainButton onClick={handlePredict}>Predict!</TrainButton>

        {predictionProb ? <p>predictionProb: {predictionProb}</p> : null}

        <div className="flex flex-row flex-wrap gap-4">
          {selectedDataset.inputs.map((input) => {
            const options = input.options.map((option, i) => ({
              id: i,
              name: option,
            }));

            const selectedOption =
              options.find((opt) => opt.name === userInputs[input.name]) ||
              options[0];

            return (
              <div key={input.name} className="flex flex-col items-center">
                <p className="text-white mb-1">{input.name}</p>
                <DropDownMenu
                  options={options}
                  value={selectedOption}
                  onChange={(option) =>
                    setUserInputs((prev) => ({
                      ...prev,
                      [input.name]: option.name,
                    }))
                  }
                />
              </div>
            );
          })}
        </div>
      </div>
    </main>
  );
}
