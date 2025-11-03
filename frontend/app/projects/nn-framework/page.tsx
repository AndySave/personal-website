"use client"

import { Button, Input } from "@headlessui/react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import DropDownMenu from "@/components/DropDownMenu";
import FeatureInputs from "@/components/FeatureInputs";
import TrainingGraph from "@/components/TrainingGraph";
import useNetworks from "@/hooks/useNetworks";
import useDatasets, { getDefaultInputs } from "@/hooks/useDatasets";
import useTraining from "@/hooks/useTraining";

export default function NeuralNetworkPage() {
  const { networks, selectedNetwork, setSelectedNetwork } = useNetworks();
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
    predictionOutput,
    trainingLoss,
    trainingAccuracy,
    handleTrain,
    handlePredict,
  } = useTraining(selectedDataset, selectedNetwork, userInputs);

  if (!datasets || !selectedDataset || !networks || !selectedNetwork) {
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
              options={networks.map((network, i) => ({
                id: i,
                value: network.display_name,
                display_name: network.display_name,
              }))}
              value={{
                id: networks.findIndex(
                  (network) =>
                    network.display_name === selectedNetwork.display_name
                ),
                value: selectedNetwork.display_name,
                display_name: selectedNetwork.display_name,
              }}
              onChange={(option) => {
                const newNetwork = networks.find(
                  (network) => network.display_name === option.display_name
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
                  (dataset) => dataset.display_name === option.display_name
                );
                if (newDataset) {
                  setSelectedDataset(newDataset);
                  setUserInputs(getDefaultInputs(newDataset));
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

        <NeuralNetworkVisualizer layerSizes={selectedNetwork.layer_sizes} />

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
          {selectedDataset.task_type !== "regression" && trainingAccuracy && (
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

          {predictionOutput && (
            <p className="text-pink-300 text-sm tracking-wide">
              Prediction output: {predictionOutput.toFixed(3)}
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
