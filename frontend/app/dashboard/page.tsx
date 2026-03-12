"use client";

import { useEffect, useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer,
} from "recharts";
import { getTrends } from "../../src/lib/api";
import type { TrendPoint } from "../../src/lib/types";
import { BRANDS, COLORS, SENTIMENT_COLORS } from "../../src/lib/constants";

function MetricCard({
  title,
  value,
  sub,
  accent,
}: {
  title: string;
  value: string;
  sub: string;
  accent: string;
}) {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: COLORS.sage }}
    >
      <p className="text-xs font-semibold uppercase tracking-wide opacity-60 mb-1">
        {title}
      </p>
      <p className="text-2xl font-bold" style={{ color: accent }}>
        {value}
      </p>
      <p className="text-xs opacity-50 mt-0.5">{sub}</p>
    </div>
  );
}

export default function DashboardPage() {
  const [brand, setBrand] = useState(BRANDS[0]);
  const [days, setDays] = useState(30);
  const [data, setData] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getTrends(brand, days)
      .then((r) => setData(r.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [brand, days]);

  const latest = data[data.length - 1];
  const totalVolume = data.reduce((s, d) => s + d.volume, 0);

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold" style={{ color: COLORS.teal }}>
          📊 Sentiment Dashboard
        </h1>
        <p className="opacity-70 mt-1">
          Real-time sentiment trends for 10 Indian brands.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div>
          <label className="block text-xs font-semibold mb-1.5 uppercase tracking-wide opacity-60">
            Brand
          </label>
          <select
            className="input-field"
            value={brand}
            onChange={(e) => setBrand(e.target.value)}
          >
            {BRANDS.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-semibold mb-1.5 uppercase tracking-wide opacity-60">
            Time Range
          </label>
          <select
            className="input-field"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
          >
            {[7, 14, 30, 60, 90].map((d) => (
              <option key={d} value={d}>{d} days</option>
            ))}
          </select>
        </div>
      </div>

      {/* Summary Metrics */}
      {latest && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Positive"
            value={`${latest.positive_pct.toFixed(1)}%`}
            sub="last data point"
            accent={COLORS.teal}
          />
          <MetricCard
            title="Neutral"
            value={`${latest.neutral_pct.toFixed(1)}%`}
            sub="last data point"
            accent={COLORS.gold}
          />
          <MetricCard
            title="Negative"
            value={`${latest.negative_pct.toFixed(1)}%`}
            sub="last data point"
            accent={COLORS.orange}
          />
          <MetricCard
            title="Total Volume"
            value={totalVolume.toLocaleString()}
            sub={`last ${days} days`}
            accent={COLORS.dark}
          />
        </div>
      )}

      {/* Area Chart */}
      <div
        className="rounded-xl p-6 shadow-card"
        style={{ background: COLORS.sage }}
      >
        <h2 className="text-lg font-semibold mb-4" style={{ color: COLORS.teal }}>
          {brand} — Sentiment Over Time
        </h2>
        {loading ? (
          <div className="h-64 flex items-center justify-center opacity-40">
            Loading…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart
              data={data}
              margin={{ top: 4, right: 16, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.mint} />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: COLORS.dark }}
                tickFormatter={(d: string) => d.slice(5)}
              />
              <YAxis
                tick={{ fontSize: 11, fill: COLORS.dark }}
                unit="%"
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  background: COLORS.dark,
                  border: "none",
                  borderRadius: 8,
                  color: COLORS.mint,
                  fontSize: 12,
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: 12, color: COLORS.dark }}
              />
              <Area
                type="monotone"
                dataKey="positive_pct"
                name="Positive %"
                stackId="1"
                stroke={SENTIMENT_COLORS.positive}
                fill={SENTIMENT_COLORS.positive}
                fillOpacity={0.7}
              />
              <Area
                type="monotone"
                dataKey="neutral_pct"
                name="Neutral %"
                stackId="1"
                stroke={SENTIMENT_COLORS.neutral}
                fill={SENTIMENT_COLORS.neutral}
                fillOpacity={0.7}
              />
              <Area
                type="monotone"
                dataKey="negative_pct"
                name="Negative %"
                stackId="1"
                stroke={SENTIMENT_COLORS.negative}
                fill={SENTIMENT_COLORS.negative}
                fillOpacity={0.7}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
