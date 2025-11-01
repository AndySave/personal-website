import { Input } from "@headlessui/react";
import { FeatureMetadata } from "@/types/nn-framework";
import DropDownMenu from "@/components/DropDownMenu";

interface Props {
  feature: FeatureMetadata;
  userInputs: Record<string, string>;
  setUserInputs: (value: React.SetStateAction<Record<string, string>>) => void;
}

export default function FeatureInputs({
  feature,
  userInputs,
  setUserInputs,
}: Props) {
  if (feature.type === "numeric") {
    return (
      <div key={feature.name} className="flex flex-col items-center">
        <p className="text-white mb-1">{feature.display_name}</p>
        <Input
          value={userInputs[feature.name]}
          type="number"
          min={feature.min}
          max={feature.max}
          className="rounded-md border border-slate-700 bg-slate-900/60 text-white hover:bg-blue-400/20 transition-all duration-200"
          onChange={(event) => {
            setUserInputs((prev) => ({
              ...prev,
              [feature.name]: event.target.value,
            }));
          }}
        />
      </div>
    );
  } else {
    if (!feature.options) return null;

    const options = feature.options.map((option, i) => ({
      id: i,
      value: option.value,
      display_name: option.display_name,
    }));

    const selectedOption =
      options.find((opt) => opt.value === userInputs[feature.name]) ||
      options[0];

    return (
      <div key={feature.name} className="flex flex-col items-center">
        <p className="text-white mb-1">{feature.display_name}</p>
        <DropDownMenu
          options={options}
          value={selectedOption}
          onChange={(option) =>
            setUserInputs((prev) => ({
              ...prev,
              [feature.name]: option.value,
            }))
          }
        />
      </div>
    );
  }
}
