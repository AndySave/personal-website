
"use client"

import {
  Listbox,
  ListboxButton,
  ListboxOption,
  ListboxOptions,
} from "@headlessui/react";
import { CheckIcon, ChevronDownIcon } from "@heroicons/react/20/solid";
import clsx from "clsx";


export interface Option {
    id: string | number;
    name: string;
}

interface Props {
    options: Option[];
    value: Option;
    onChange: (value: Option) => void;
    label?: string;
    disabled?: boolean;
    className?: string;
}


export default function DropDownMenu({
  options,
  value,
  onChange,
  label,
  disabled = false,
  className = "",
}: Props) {
  return (
    <div className={clsx("mx-auto w-52", className)}>
      {label && (
        <p className="mb-1 text-sm font-medium text-gray-300">{label}</p>
      )}

      <Listbox value={value} onChange={onChange} disabled={disabled}>
        <ListboxButton
          className={clsx(
            "relative block w-full rounded-lg bg-white/5 py-1.5 pr-8 pl-3 text-left text-sm/6 text-white",
            "focus:outline-none data-focus:outline-2 data-focus:-outline-offset-2 data-focus:outline-white/25",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          {value?.name ?? "Select..."}
          <ChevronDownIcon
            className="group pointer-events-none absolute top-2.5 right-2.5 size-4 fill-white/60"
            aria-hidden="true"
          />
        </ListboxButton>

        <ListboxOptions
          anchor="bottom"
          transition
          className={clsx(
            "w-(--button-width) rounded-xl border border-white/5 bg-white/5 p-1 [--anchor-gap:--spacing(1)] focus:outline-none",
            "transition duration-100 ease-in data-leave:data-closed:opacity-0"
          )}
        >
          {options.map((opt) => (
            <ListboxOption
              key={opt.id}
              value={opt}
              className="group flex cursor-default items-center gap-2 rounded-lg px-3 py-1.5 select-none data-focus:bg-white/10"
            >
              <CheckIcon className="invisible size-4 fill-white group-data-selected:visible" />
              <div className="text-sm/6 text-white">{opt.name}</div>
            </ListboxOption>
          ))}
        </ListboxOptions>
      </Listbox>
    </div>
  );
}
