# Defense Spend Atlas

A lightweight front-end app that uses the USAspending API to:

- Track federal contract obligations for military contractors.
- Visualize top locations where obligated dollars are spent.
- Forecast next-fiscal-year contract revenue using a simple linear trend model.

## Run locally

Because this is a static web app, you can run it with any static server.

```bash
python3 -m http.server 8080
```

Then open <http://localhost:8080>.

## Notes

- Data comes from `https://api.usaspending.gov/api/v2`.
- The forecast is a directional estimate and not financial advice.
