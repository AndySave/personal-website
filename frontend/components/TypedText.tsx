
"use client";

import { ReactTyped } from "react-typed";

type TypedProps = {
  strings: string[];
  typeSpeed?: number;
  backSpeed?: number;
  loop?: boolean;
  className?: string;
};

export default function TypedText({
  strings,
  typeSpeed = 60,
  backSpeed = 80,
  loop = true,
  className,
}: TypedProps) {
  return (
    <span className={className}>
      <ReactTyped strings={strings} typeSpeed={typeSpeed} backSpeed={backSpeed} loop={loop}/>
    </span>
  );
}
