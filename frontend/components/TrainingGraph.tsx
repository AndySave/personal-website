"use client";

import { LineChart } from "@mui/x-charts";
import { axisClasses } from "@mui/x-charts/ChartsAxis";


interface Props {
    title: string
    xAxixLabel: string
    yAxisLabel: string
    data: number[]
}


export default function TrainingGraph({title, xAxixLabel, yAxisLabel, data}: Props) {
  return (
    <LineChart
      series={[
        {
          data: data,
          label: title,
          showMark: false,
          color: "#7c0e8f",
          curve: "monotoneX",
        },
      ]}
      xAxis={[
        {
          data: Array.from({ length: data.length }, (_, i) => i + 1),
          label: xAxixLabel,
        },
      ]}
      yAxis={[{ label: yAxisLabel, width: 40 }]}
      height={320}
      margin={{ right: 50 }}
      slotProps={{
        legend: {
          sx: {
            color: "#ffffff",
          },
        },
      }}
      sx={{
        borderRadius: 8,
        border: "2px solid rgba(100,116,139,0.3)",

        // axis lines
        [`& .${axisClasses.left} .${axisClasses.line}`]: {
          stroke: "#64748b",
        },
        [`& .${axisClasses.bottom} .${axisClasses.line}`]: {
          stroke: "#64748b",
        },

        // axis ticks
        [`& .${axisClasses.left} .${axisClasses.tick}`]: {
          stroke: "#64748b",
        },
        [`& .${axisClasses.bottom} .${axisClasses.tick}`]: {
          stroke: "#64748b",
        },

        // tick labels
        [`& .${axisClasses.left} .${axisClasses.tickLabel}`]: {
          fill: "#cbd5e1",
        },
        [`& .${axisClasses.bottom} .${axisClasses.tickLabel}`]: {
          fill: "#cbd5e1",
        },

        // axis titles
        [`& .${axisClasses.left} .${axisClasses.label}`]: {
          fill: "#ffffff",
          fontWeight: 500,
        },
        [`& .${axisClasses.bottom} .${axisClasses.label}`]: {
          fill: "#ffffff",
          fontWeight: 500,
        },
      }}
    />
  );
}
