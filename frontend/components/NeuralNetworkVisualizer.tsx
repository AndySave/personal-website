
import { renderNeuralNetworkSVG, type NNProps } from "@/lib/nnLayout";

export default function NeuralNetworkVisualizer(props: NNProps) {
  return (
    <div className="w-fit rounded-2xl border border-slate-700 bg-slate-900/60 p-5 shadow-2xl backdrop-blur">
      {renderNeuralNetworkSVG(props)}
    </div>
  );
}
