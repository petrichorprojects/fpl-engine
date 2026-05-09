"use client";
import SquadBuilder from "@/components/SquadBuilder";

export default function SquadPage() {
  return (
    <div className="max-w-6xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Squad Optimiser</h1>
      <p className="text-sm text-gray-400 mb-6">
        MILP optimizer with gamestate-adaptive differential weighting
      </p>
      <SquadBuilder />
    </div>
  );
}
