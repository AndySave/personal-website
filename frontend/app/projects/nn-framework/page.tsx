"use client";

import { useState } from "react";
import NeuralNetworkVisualizer from "@/components/NeuralNetworkVisualizer";
import NetworkDropDown, { Network } from "@/components/NetworkDropDown";
import DatasetDropDown, {Dataset} from "@/components/DatasetDropDown";

const NETWORKS: Network[] = [
  { id: 1, name: "Small network", layers: [2, 4, 1] },
  { id: 2, name: "Medium network", layers: [2, 8, 4, 1] },
  { id: 3, name: "Large network", layers: [2, 12, 12, 4, 1] },
];

const DATASETS: Dataset[] = [
  { id: 1, name: "Dataset 1" },
  { id: 2, name: "Dataset 2" },
  { id: 3, name: "Dataset 3" },
];

export default function NeuralNetworkPage() {
  const [selectedNetwork, setSelectedNetwork] = useState<Network>(NETWORKS[0]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset>(DATASETS[0]);

  return (
    <main className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col justify-center items-center p-20">
        <div className="flex pb-5 gap-10">
          <NetworkDropDown
            options={NETWORKS}
            value={selectedNetwork}
            onChange={setSelectedNetwork}
          />
          <DatasetDropDown options={DATASETS} dataset={selectedDataset} onChange={setSelectedDataset}/>
        </div>
        <NeuralNetworkVisualizer layerSizes={selectedNetwork.layers} />
      </div>
    </main>
  );
}
