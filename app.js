const API_BASE = "https://api.usaspending.gov/api/v2";
const recipientInput = document.getElementById("recipientInput");
const startYearInput = document.getElementById("startYearInput");
const endYearInput = document.getElementById("endYearInput");
const loadBtn = document.getElementById("loadBtn");

const totalEl = document.getElementById("totalObligation");
const avgEl = document.getElementById("avgAnnual");
const forecastEl = document.getElementById("nextYearForecast");
const forecastNoteEl = document.getElementById("forecastNote");
const highlightsEl = document.getElementById("highlights");

const currentYear = new Date().getUTCFullYear();
startYearInput.value = currentYear - 5;
endYearInput.value = currentYear;
recipientInput.value = "Lockheed Martin";

const map = L.map("map", { worldCopyJump: true }).setView([39.5, -98.35], 4);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const markersLayer = L.layerGroup().addTo(map);

loadBtn.addEventListener("click", () => {
  loadData().catch((error) => {
    console.error(error);
    alert("Unable to load USAspending data right now.");
  });
});

function formatMoney(amount) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(amount || 0);
}

function buildSearchBody() {
  const startFy = Number(startYearInput.value);
  const endFy = Number(endYearInput.value);

  return {
    filters: {
      time_period: [
        {
          start_date: `${startFy - 1}-10-01`,
          end_date: `${endFy}-09-30`,
        },
      ],
      award_type_codes: ["A", "B", "C", "D"],
      recipient_search_text: [recipientInput.value.trim()],
    },
  };
}

async function fetchSummary() {
  const body = {
    ...buildSearchBody(),
    category: "naics",
    limit: 1,
    page: 1,
  };

  const response = await fetch(`${API_BASE}/search/spending_by_category/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Summary fetch failed: ${response.status}`);
  }
  return response.json();
}

async function fetchByState() {
  const body = {
    ...buildSearchBody(),
    category: "place_of_performance",
    limit: 50,
    page: 1,
  };

  const response = await fetch(`${API_BASE}/search/spending_by_category/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`State fetch failed: ${response.status}`);
  }
  return response.json();
}

async function fetchByYear() {
  const body = {
    ...buildSearchBody(),
    group: "fiscal_year",
    limit: 100,
    page: 1,
  };

  const response = await fetch(`${API_BASE}/search/spending_over_time/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Year fetch failed: ${response.status}`);
  }
  return response.json();
}

function projectNextYear(points) {
  const ordered = [...points].sort((a, b) => a.x - b.x);
  if (ordered.length < 2) {
    return ordered[0]?.y ?? 0;
  }

  const n = ordered.length;
  const meanX = ordered.reduce((sum, p) => sum + p.x, 0) / n;
  const meanY = ordered.reduce((sum, p) => sum + p.y, 0) / n;

  let numerator = 0;
  let denominator = 0;
  for (const p of ordered) {
    const dx = p.x - meanX;
    numerator += dx * (p.y - meanY);
    denominator += dx * dx;
  }

  const slope = denominator === 0 ? 0 : numerator / denominator;
  const intercept = meanY - slope * meanX;
  return Math.max(0, intercept + slope * (ordered[n - 1].x + 1));
}

function drawTrend(points, forecastValue) {
  const canvas = document.getElementById("trendCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = 320;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);
  if (!points.length) {
    return;
  }

  const sorted = [...points].sort((a, b) => a.x - b.x);
  const values = sorted.map((p) => p.y);
  values.push(forecastValue);

  const max = Math.max(...values) * 1.1;
  const min = 0;
  const pad = 30;
  const stepX = (width - pad * 2) / (sorted.length);

  function yToCanvas(v) {
    return height - pad - ((v - min) / (max - min || 1)) * (height - pad * 2);
  }

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + ((height - 2 * pad) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  sorted.forEach((point, i) => {
    const x = pad + stepX * i;
    const y = yToCanvas(point.y);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = "#0f766e";
  ctx.beginPath();
  const lastX = pad + stepX * (sorted.length - 1);
  const lastY = yToCanvas(sorted[sorted.length - 1].y);
  const forecastX = pad + stepX * sorted.length;
  const forecastY = yToCanvas(forecastValue);
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(forecastX, forecastY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#1f2937";
  ctx.font = "12px Inter";
  sorted.forEach((point, i) => {
    const x = pad + stepX * i;
    ctx.fillText(String(point.x), x - 12, height - 8);
  });
  ctx.fillText(`${sorted[sorted.length - 1].x + 1}f`, forecastX - 10, height - 8);
}

const stateCoordinates = {
  AL: [32.806671, -86.79113], AK: [61.370716, -152.404419], AZ: [33.729759, -111.431221],
  AR: [34.969704, -92.373123], CA: [36.116203, -119.681564], CO: [39.059811, -105.311104],
  CT: [41.597782, -72.755371], DE: [39.318523, -75.507141], FL: [27.766279, -81.686783],
  GA: [33.040619, -83.643074], HI: [21.094318, -157.498337], ID: [44.240459, -114.478828],
  IL: [40.349457, -88.986137], IN: [39.849426, -86.258278], IA: [42.011539, -93.210526],
  KS: [38.5266, -96.726486], KY: [37.66814, -84.670067], LA: [31.169546, -91.867805],
  ME: [44.693947, -69.381927], MD: [39.063946, -76.802101], MA: [42.230171, -71.530106],
  MI: [43.326618, -84.536095], MN: [45.694454, -93.900192], MS: [32.741646, -89.678696],
  MO: [38.456085, -92.288368], MT: [46.921925, -110.454353], NE: [41.12537, -98.268082],
  NV: [38.313515, -117.055374], NH: [43.452492, -71.563896], NJ: [40.298904, -74.521011],
  NM: [34.840515, -106.248482], NY: [42.165726, -74.948051], NC: [35.630066, -79.806419],
  ND: [47.528912, -99.784012], OH: [40.388783, -82.764915], OK: [35.565342, -96.928917],
  OR: [44.572021, -122.070938], PA: [40.590752, -77.209755], RI: [41.680893, -71.51178],
  SC: [33.856892, -80.945007], SD: [44.299782, -99.438828], TN: [35.747845, -86.692345],
  TX: [31.054487, -97.563461], UT: [40.150032, -111.862434], VT: [44.045876, -72.710686],
  VA: [37.769337, -78.169968], WA: [47.400902, -121.490494], WV: [38.491226, -80.954453],
  WI: [44.268543, -89.616508], WY: [42.755966, -107.30249], DC: [38.9072, -77.0369],
};

function drawMap(rows) {
  markersLayer.clearLayers();

  const filtered = rows
    .map((row) => {
      const code = row.code?.toUpperCase();
      return {
        ...row,
        code,
        coords: stateCoordinates[code],
      };
    })
    .filter((row) => row.coords && row.amount > 0)
    .sort((a, b) => b.amount - a.amount)
    .slice(0, 12);

  filtered.forEach((row) => {
    const radius = Math.max(6, Math.min(28, Math.sqrt(row.amount) / 3500));
    L.circleMarker(row.coords, {
      radius,
      color: "#1d4ed8",
      fillColor: "#3b82f6",
      fillOpacity: 0.55,
      weight: 1,
    })
      .bindPopup(`<strong>${row.name || row.code}</strong><br/>${formatMoney(row.amount)}`)
      .addTo(markersLayer);
  });
}

function renderHighlights(company, byYear, byState, forecast) {
  const sortedYears = [...byYear].sort((a, b) => b.y - a.y);
  const sortedStates = [...byState].sort((a, b) => b.amount - a.amount);
  const topYear = sortedYears[0];
  const topState = sortedStates[0];

  highlightsEl.innerHTML = "";

  const lines = [
    `${company} reached a peak observed fiscal year obligation of ${formatMoney(topYear?.y || 0)} in FY ${topYear?.x || "N/A"}.`,
    `${topState?.name || "Top location"} captured ${formatMoney(topState?.amount || 0)} in obligated dollars.`,
    `Model-based next fiscal year projection: ${formatMoney(forecast)} based on linear trend in observed annual obligations.`,
    "Forecasts are directional estimates and should be validated against awarded backlog, options, and agency budget outlooks.",
  ];

  lines.forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    highlightsEl.appendChild(li);
  });
}

async function loadData() {
  loadBtn.disabled = true;
  loadBtn.textContent = "Loading…";

  const [summary, byState, byYear] = await Promise.all([
    fetchSummary(),
    fetchByState(),
    fetchByYear(),
  ]);

  const annual = (byYear.results || [])
    .map((row) => ({
      x: Number(row.time_period?.fiscal_year),
      y: Number(row.aggregated_amount || 0),
    }))
    .filter((row) => Number.isFinite(row.x));

  const locations = (byState.results || []).map((row) => ({
    name: row.name,
    code: row.code,
    amount: Number(row.amount || 0),
  }));

  const totalObligation = Number(summary?.page_metadata?.total || 0);
  const avgAnnual = annual.length
    ? annual.reduce((sum, p) => sum + p.y, 0) / annual.length
    : 0;
  const projectedNext = projectNextYear(annual);

  totalEl.textContent = formatMoney(totalObligation);
  avgEl.textContent = formatMoney(avgAnnual);
  forecastEl.textContent = formatMoney(projectedNext);
  forecastNoteEl.textContent = `Observed FY window: ${startYearInput.value}–${endYearInput.value}`;

  drawMap(locations);
  drawTrend(annual, projectedNext);
  renderHighlights(recipientInput.value.trim(), annual, locations, projectedNext);

  loadBtn.disabled = false;
  loadBtn.textContent = "Load data";
}

loadData().catch((error) => {
  console.error(error);
  loadBtn.disabled = false;
  loadBtn.textContent = "Load data";
});
