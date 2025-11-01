"use client";

import { useState } from "react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import NetworkDropDown, { Network } from "@/components/NetworkDropDown";
import TrainButton from "./TrainButton";
import DropDownMenu from "@/components/DropDownMenu";
import FeatureInputs from "@/components/FeatureInputs";
import useDatasets from "@/hooks/useDatasets";
import useTraining from "@/hooks/useTraining";
import { DatasetMetadata } from "@/types/nn-framework";


const NETWORKS: Network[] = [
  {
    id: 1,
    name: "Small network",
    layers: [
      { type: "dense", inFeatures: 30, outFeatures: 4 },
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
      { type: "dense", inFeatures: 30, outFeatures: 8 },
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

const getLayerSizes = (network: Network, dataset: DatasetMetadata) => {
  const layerSizes = [dataset.features.length];

  for (const layer of network.layers) {
    if (layer.outFeatures !== undefined) {
      layerSizes.push(layer.outFeatures);
    }
  }

  return layerSizes;
};


export default function NeuralNetworkPage() {
  const [selectedNetwork, setSelectedNetwork] = useState<Network>(NETWORKS[0]);
  const {datasets, selectedDataset, setSelectedDataset, userInputs, setUserInputs} = useDatasets();
  const {isTraining, accuracy, predictionProb, handleTrain, handlePredict} = useTraining(selectedDataset, selectedNetwork, userInputs);

  if (!datasets || !selectedDataset) {
    return <p>Loading...</p>;
  }

  return (
    <main className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col justify-center items-center p-20">
        <div className="flex pb-5 gap-10">
          <NetworkDropDown
            options={NETWORKS}
            value={selectedNetwork}
            onChange={setSelectedNetwork}
          />

          <DropDownMenu
            options={datasets.map((dataset, i) => ({
              id: i,
              value: dataset.name,
              display_name: dataset.display_name,
            }))}
            value={{
              id: datasets.findIndex(
                (dataset) => dataset.name === selectedDataset.name
              ),
              value: selectedDataset.name,
              display_name: selectedDataset.display_name,
            }}
            onChange={(option) => {
              const newDataset = datasets.find(
                (dataset) => dataset.name === option.display_name
              );
              if (newDataset) {
                setSelectedDataset(newDataset);
              }
            }}
          />
        </div>
        {
          <NeuralNetworkVisualizer
            layerSizes={getLayerSizes(selectedNetwork, selectedDataset)}
          />
        }
        <TrainButton onClick={handleTrain}>Train!</TrainButton>
        <TrainButton onClick={handlePredict}>Predict!</TrainButton>

        {predictionProb ? <p>predictionProb: {predictionProb}</p> : null}

        <div className="flex flex-row flex-wrap gap-4">
          {selectedDataset?.features.map((feature) => (
            <FeatureInputs
              key={feature.name}
              feature={feature}
              userInputs={userInputs}
              setUserInputs={setUserInputs}
            />
          ))}
        </div>
      </div>
    </main>
  );
}
