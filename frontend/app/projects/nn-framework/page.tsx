"use client";

import { useState } from "react";
import { Button, Input } from "@headlessui/react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import DropDownMenu from "@/components/DropDownMenu";
import FeatureInputs from "@/components/FeatureInputs";
import TrainingGraph from "@/components/TrainingGraph";
import useDatasets from "@/hooks/useDatasets";
import useTraining from "@/hooks/useTraining";
import { DatasetMetadata, Network } from "@/types/nn-framework";

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
  const {
    datasets,
    selectedDataset,
    setSelectedDataset,
    userInputs,
    setUserInputs,
  } = useDatasets();
  const {
    epochs,
    setEpochs,
    isTraining,
    predictionProb,
    trainingLoss,
    trainingAccuracy,
    handleTrain,
    handlePredict,
  } = useTraining(selectedDataset, selectedNetwork, userInputs);

  if (!datasets || !selectedDataset) {
    return <p>Loading...</p>;
  }

  return (
    <main className="flex flex-col items-center min-h-screen w-full px-8 py-16 text-slate-200 space-y-24">
      <section className="flex flex-col items-center w-full max-w-5xl space-y-10">
        <h2 className="text-slate-400 tracking-widest text-sm">
          MODEL TRAINING
        </h2>

        <div className="flex flex-wrap justify-center gap-8">
          <div className="flex flex-col items-center">
            <h2>Networks</h2>
            <DropDownMenu
              options={NETWORKS.map((network, i) => ({
                id: i,
                value: network.name,
                display_name: network.name,
              }))}
              value={{
                id: NETWORKS.findIndex(
                  (network) => network.name === selectedNetwork.name
                ),
                value: selectedNetwork.name,
                display_name: selectedNetwork.name,
              }}
              onChange={(option) => {
                const newNetwork = NETWORKS.find(
                  (network) => network.name === option.display_name
                );
                if (newNetwork) {
                  setSelectedNetwork(newNetwork);
                }
              }}
            />
          </div>

          <div className="flex flex-col items-center">
            <h2>Datasets</h2>
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

          <div className="flex flex-col items-center">
            <h2>Epochs</h2>
            <Input
              value={epochs}
              type="number"
              min={1}
              max={500}
              className="rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
              onChange={(event) => setEpochs(event.target.value)}
            />
          </div>
        </div>

        <NeuralNetworkVisualizer
          layerSizes={getLayerSizes(selectedNetwork, selectedDataset)}
        />

        <Button
          onClick={handleTrain}
          disabled={isTraining}
          className={
            "px-6 py-2 rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
          }
        >
          {isTraining ? "Training..." : "Train"}
        </Button>

        <div className="flex flex-wrap gap-10">
          {trainingLoss && (
            <TrainingGraph
              title="Training Loss"
              xAxixLabel="Epoch"
              yAxisLabel="Loss"
              data={trainingLoss}
            />
          )}
          {trainingAccuracy && (
            <TrainingGraph
              title="Training Accuracy"
              xAxixLabel="Epoch"
              yAxisLabel=""
              data={trainingAccuracy}
            />
          )}
        </div>
      </section>

      {/* Divider line */}
      <div className="w-full h-px bg-linear-to-r from-transparent via-slate-700/50 to-transparent" />

      <section className="flex flex-col items-center w-full max-w-5xl space-y-8">
        <h2 className="text-slate-400 tracking-widest text-sm">
          MAKE A PREDICTION
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 w-full">
          {selectedDataset?.features.map((feature) => (
            <FeatureInputs
              key={feature.name}
              feature={feature}
              userInputs={userInputs}
              setUserInputs={setUserInputs}
            />
          ))}
        </div>

        <div className="flex flex-col items-center space-y-3">
          <Button
            onClick={handlePredict}
            className={
              "px-6 py-2 rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
            }
          >
            Predict
          </Button>

          {predictionProb && (
            <p className="text-pink-300 text-sm tracking-wide">
              Prediction probability: {predictionProb.toFixed(3)}
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
