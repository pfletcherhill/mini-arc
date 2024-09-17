import { COLORS, Grid } from "@/types";
import React from "react";

interface GridProps {
  grid: Grid;
}

const GridStatic: React.FC<GridProps> = ({ grid }) => {
  return (
    <div className="inline-block border border-gray-500">
      {grid.map((row, rowIndex) => (
        <div key={rowIndex} className="flex">
          {row.map((cell, cellIndex) => (
            <div
              key={cellIndex}
              className="w-6 h-6 border-0.5 border-gray-500"
              style={{
                backgroundColor: COLORS[cell],
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

export default GridStatic;
