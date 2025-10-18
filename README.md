# BLN-AQ

A study concenring the interaction and spatio-temporal modeling of various features on the air quality of Berlin.

### Goals
- Model urban air pollution levels.
- Quantify drivers and correlations of pollution spikes
- Provide insights for stakeholders (e.g. "these areas experience an increase by 20% of NO2 in winter at 3pm")

### Deliverables
1. Poster for final presentation
2. An interactive folium map
3. A python notebook with exploratory code

### Data
- Berlin air quality: [Collection Endpoint](https://luftdaten.berlin.de/pollution/)
- Berlin weather: [Index](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/) [Stations](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/TU_Stundenwerte_Beschreibung_Stationen.txt)
- Traffic: []() todo!

### Methods
- EDA -> Folium heatmap with slider for time & various layers
- Correlation (w/ lags + auto) between time series
- Granger causality tests
- Extreme value theory (from Kimi!)
