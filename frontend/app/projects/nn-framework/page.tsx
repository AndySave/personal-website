"use client";

import { Button, Input } from "@headlessui/react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import DropDownMenu from "@/components/DropDownMenu";
import FeatureInputs from "@/components/FeatureInputs";
import TrainingGraph from "@/components/TrainingGraph";
import useNetworks from "@/hooks/useNetworks";
import useDatasets, { getDefaultInputs } from "@/hooks/useDatasets";
import useTraining from "@/hooks/useTraining";
import { useRef, useEffect } from "react";

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
    activations,
    trainingLoss,
    trainingAccuracy,
    handleTrain,
    handlePredict,
  } = useTraining(selectedDataset, selectedNetwork, userInputs);

  const predictionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (predictionOutput && predictionRef.current) {
      predictionRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [predictionOutput]);

  if (!datasets || !selectedDataset || !networks || !selectedNetwork) {
    return <p className="text-center mt-10 text-slate-400">Loading...</p>;
  }

  return (
    <main className="flex flex-col items-center min-h-screen w-full px-8 py-16 text-slate-200 space-y-14">
      {/* STEP 1: TRAINING SECTION */}
      <section
        id="training"
        className="flex flex-col items-center w-full max-w-5xl bg-slate-800/30 border border-slate-700/50 rounded-xl p-10 space-y-10 backdrop-blur-sm"
      >
        <h3 className="flex items-center gap-3 text-slate-400 tracking-widest text-sm uppercase">
          <span className="w-6 h-6 flex items-center justify-center rounded-full border border-slate-500 text-xs">
            1
          </span>
          Model Training
        </h3>

        <p className="text-slate-400 text-sm text-center max-w-2xl">
          {selectedDataset.description}
        </p>

        {/* Controls */}
        <div className="flex flex-wrap justify-center gap-8">
          <div className="flex flex-col items-center space-y-2">
            <h2 className="text-sm text-slate-300">Networks</h2>
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
                if (newNetwork) setSelectedNetwork(newNetwork);
              }}
            />
          </div>

          <div className="flex flex-col items-center space-y-2">
            <h2 className="text-sm text-slate-300">Datasets</h2>
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

          <div className="flex flex-col items-center space-y-2">
            <h2 className="text-sm text-slate-300">Epochs</h2>
            <Input
              value={epochs}
              type="number"
              min={1}
              max={500}
              className="w-full rounded-md border border-slate-700 bg-slate-900/60 text-white text-center hover:bg-blue-400/20 transition-all duration-200"
              onChange={(event) => setEpochs(event.target.value)}
            />
          </div>
        </div>

        {/* Train button */}
        <Button
          onClick={handleTrain}
          disabled={isTraining}
          className="px-6 py-2 rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
        >
          {isTraining ? (
            <p className="text-xs animate-pulse">
              Training in progress — this may take a few seconds...
            </p>
          ) : (
            "Train"
          )}
        </Button>

        {/* Training Graphs */}
        <div
          className="flex flex-wrap justify-center gap-10 transition-opacity duration-500 ease-in"
          style={{ opacity: trainingLoss ? 1 : 0 }}
        >
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
              yAxisLabel="Accuracy"
              data={trainingAccuracy}
            />
          )}
        </div>

        {/* Training Summary */}
        {trainingLoss && (
          <p className="text-sm text-slate-400 text-center">
            Final loss: {trainingLoss[trainingLoss.length - 1].toFixed(3)}
            {trainingAccuracy && (
              <>
                {" "}
                | Accuracy:{" "}
                {trainingAccuracy[trainingAccuracy.length - 1].toFixed(3)}
              </>
            )}
          </p>
        )}
      </section>

      {/* Divider */}
      <div className="w-full h-px bg-linear-to-r from-transparent via-slate-700/50 to-transparent" />

      {/* STEP 2: PREDICTION SECTION */}
      <section
        id="prediction"
        ref={predictionRef}
        className="flex flex-col items-center w-full max-w-5xl bg-slate-800/30 border border-slate-700/50 rounded-xl p-10 space-y-10 backdrop-blur-sm"
      >
        <h3 className="flex items-center gap-3 text-slate-400 tracking-widest text-sm uppercase">
          <span className="w-6 h-6 flex items-center justify-center rounded-full border border-slate-500 text-xs">
            2
          </span>
          Make a Prediction
        </h3>

        <p className="text-slate-400 text-sm italic text-center max-w-xl">
          Enter feature values to generate a prediction using the trained model.
        </p>

        {/* Feature Inputs */}
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

        {/* Predict Button + Output */}
        <div className="flex flex-col items-center space-y-3">
          <Button
            onClick={handlePredict}
            disabled={isTraining}
            className="px-6 py-2 rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
          >
            Predict
          </Button>

          {predictionOutput && (
            <p className="text-pink-300 text-sm tracking-wide transition-opacity duration-500">
              Prediction output: {predictionOutput.toFixed(3)}
            </p>
          )}

          {activations && selectedDataset && (
            <NeuralNetworkVisualizer
              activations={activations}
              features={selectedDataset.features}
            />
          )}
        </div>
      </section>
    </main>
  );
}
