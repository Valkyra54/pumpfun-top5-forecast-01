# pumpfun-top5-forecast-01

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/Valkyra54/pumpfun-top5-forecast-01/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/Valkyra54/pumpfun-top5-forecast-01/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A momentum / lifecycle forecaster for pump.fun coins: it fits age-banded drift and volatility from the snapshot history, Monte-Carlo simulates each candidate coin to every answer read, deflates each place per horizon by a newcomer hazard fitted from the recent mark pairs, and sizes each place lognormally around the simulated place caps — wide, robust dispersion, with the center blended toward a heavily damped empirical place-cap drift, and the lower-place centers anchored just below place 1's (the top 5 caps run tightly compressed, so the Monte Carlo's spread-out lower places would sit far below the truth) — leaving room for the coins born after the cut.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=nbgfvtz4&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
