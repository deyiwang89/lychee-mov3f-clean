# K230 hardware summary

`k230_measurements.csv` records the frozen seed-43 PTQ artifact under the tested CanMV K230 setup. The run lasted 60 minutes with no observed errors, restarts, or interruptions. The reported 4.75 V and 0.66 A are long-time averages at the board input, giving 3.135 W (reported as 3.14 W) estimated system-level input power. The maximum surface temperature visible in the captured thermal image was 39.1 degC.

These values are operating-condition evidence for this board, firmware, model, and measurement path. They are not chip-only power, and they do not establish deployment-accuracy superiority. The PTQ accuracy degradation and unavailable peak memory/CPU/KPU utilization are documented in the local submission package.
