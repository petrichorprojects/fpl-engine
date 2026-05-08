"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  gameweek: number;
  [strategy: string]: number;
}

interface Props {
  data: DataPoint[];
  strategies: string[];
}

const COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

export default function PointsChart({ data, strategies }: Props) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
        No backtest data yet. Run a backtest first.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis
          dataKey="gameweek"
          tick={{ fontSize: 11, fill: "#9ca3af" }}
          label={{ value: "Gameweek", position: "insideBottom", offset: -2, fontSize: 11, fill: "#9ca3af" }}
        />
        <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
          labelFormatter={(v) => `GW ${v}`}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {strategies.map((s, i) => (
          <Line
            key={s}
            type="monotone"
            dataKey={s}
            stroke={COLORS[i % COLORS.length]}
            dot={false}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
