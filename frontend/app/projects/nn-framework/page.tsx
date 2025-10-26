"use client";
import { useMemo, useState } from "react";

import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'

type NodePoint = { x: number; y: number; u: number };

const DEFAULT_LAYER_SIZES = [4, 4, 8, 3, 1];
const CANVAS_WIDTH = 900;
const CANVAS_HEIGHT = 520;
const NODE_RADIUS = 14;
const PADDING = 40;

/** Parse a JSON array of positive integers; fallback to default. */
const parseLayerSizes = (rawText: string): number[] => {
  try {
    const candidate = JSON.parse(rawText.replace(/'/g, '"'));
    return Array.isArray(candidate) &&
      candidate.every((n) => Number.isInteger(n) && n > 0)
      ? candidate
      : DEFAULT_LAYER_SIZES;
  } catch {
    return DEFAULT_LAYER_SIZES;
  }
};

/** Layout nodes in columns; each point gets a vertical fraction u∈[0,1]. */
const layoutLayers = (layerSizes: number[]): NodePoint[][] => {
  const columnCount = layerSizes.length;
  const innerWidth = CANVAS_WIDTH - 2 * PADDING;
  const innerHeight = CANVAS_HEIGHT - 2 * PADDING;

  const xAtColumn = (columnIndex: number) =>
    PADDING +
    (columnCount === 1 ? 0 : (columnIndex / (columnCount - 1)) * innerWidth);

  return layerSizes.map((nodeCount, columnIndex) =>
    Array.from({ length: nodeCount }, (_, nodeIndex) => {
      const u = nodeCount === 1 ? 0.5 : nodeIndex / (nodeCount - 1);
      return {
        x: xAtColumn(columnIndex),
        y: PADDING + ((nodeIndex + 0.5) / nodeCount) * innerHeight,
        u,
      };
    })
  );
};

const path = (fromNode: NodePoint, toNode: NodePoint): string =>
  `M${fromNode.x},${fromNode.y} L${toNode.x},${toNode.y}`;

export default function NeuralNetworkPage() {
  const [inputText, setInputText] = useState(
    JSON.stringify(DEFAULT_LAYER_SIZES)
  );
  const layerSizes = useMemo(() => parseLayerSizes(inputText), [inputText]);
  const layerColumns = useMemo(() => layoutLayers(layerSizes), [layerSizes]);

  return (
    <main className="min-h-screen flex flex-col items-center">
      <div className="py-8 text-center">
        <h1 className="text-3xl font-semibold">Neural Network Visualizer</h1>
        <input
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="mt-3 rounded-md bg-slate-900 px-3 py-2 text-sm font-mono ring-1 ring-slate-700 focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      <div className="relative">
        <div className="absolute inset-0 blur-3xl opacity-40 pointer-events-none">
          <div className="h-64 w-full bg-linear-to-r from-indigo-500 via-violet-500 to-fuchsia-500 rounded-full" />
        </div>

        <div className="relative rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-2xl backdrop-blur">
          <svg
            viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
          >
            <defs>
              <linearGradient
                id="wire"
                gradientUnits="userSpaceOnUse"
                x1="0"
                y1="0"
                x2={CANVAS_WIDTH}
                y2="0"
              >
                <stop offset="0%" stopColor="#6366f1" stopOpacity="0.45" />
                <stop offset="100%" stopColor="#ec4899" stopOpacity="0.45" />
              </linearGradient>
              <radialGradient id="node" cx="50%" cy="50%" r="60%">
                <stop offset="0%" stopColor="white" />
                <stop offset="70%" stopColor="#818cf8" />
                <stop offset="100%" stopColor="#1e293b" />
              </radialGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Edges */}
            <g>
              {layerColumns
                .slice(0, -1)
                .map((column, columnIndex) =>
                  column.flatMap((sourceNode, sourceIndex) =>
                    layerColumns[columnIndex + 1].map(
                      (targetNode, targetIndex) => (
                        <path
                          key={`e-${columnIndex}-${sourceIndex}-${targetIndex}`}
                          d={path(sourceNode, targetNode)}
                          stroke="url(#wire)"
                          strokeWidth={1.5}
                        />
                      )
                    )
                  )
                )}
            </g>

            {/* Nodes */}
            <g filter="url(#glow)">
              {layerColumns.map((column, columnIndex) =>
                column.map((pt, nodeIndex) => (
                  <circle
                    key={`n-${columnIndex}-${nodeIndex}`}
                    cx={pt.x}
                    cy={pt.y}
                    r={NODE_RADIUS}
                    fill="url(#node)"
                  />
                ))
              )}
            </g>
          </svg>
        </div>
      </div>
    </main>
  );
}
