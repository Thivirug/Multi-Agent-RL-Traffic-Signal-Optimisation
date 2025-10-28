### System-Wide Metrics
These columns provide statistics for the entire traffic network.

*   **`step`**: The current time step of the simulation, usually in seconds.
*   **`system_total_running`**: The total number of vehicles currently moving in the simulation.
*   **`system_total_backlogged`**: The total number of vehicles that are queued or unable to move as desired.
*   **`system_total_stopped`**: The total number of vehicles that are currently stationary (speed is close to zero).
*   **`system_total_arrived`**: The cumulative number of vehicles that have reached their destination.
*   **`system_total_departed`**: The cumulative number of vehicles that have entered the simulation.
*   **`system_total_teleported`**: The number of vehicles removed from the simulation, often because they were stuck for too long. A high number can indicate gridlock.
*   **`system_total_waiting_time`**: The cumulative sum of waiting time for all vehicles in the system. 
*   **`system_mean_waiting_time`**: The average waiting time per vehicle across the entire system.
*   **`system_mean_speed`**: The average speed of all vehicles in the system.

### Per-Agent (Intersection) Metrics
These columns provide statistics for specific traffic light agents (intersections), identified by the number at the beginning of the column name (e.g., `1`, `2`, `5`, `6`).

*   **`[id]_stopped`**: The number of vehicles currently stopped at the intersection controlled by agent `[id]`.
*   **`[id]_accumulated_waiting_time`**: The cumulative waiting time of all vehicles that have passed through the intersection controlled by agent `[id]`.
*   **`[id]_average_speed`**: The average speed of vehicles at the intersection controlled by agent `[id]`.

### Aggregated Agent Metrics
These columns provide a sum of the metrics across all the individual agents being tracked.

*   **`agents_total_stopped`**: The total number of vehicles stopped across all monitored intersections. This is likely the sum of all `[id]_stopped` columns.
*   **`agents_total_accumulated_waiting_time`**: The total accumulated waiting time across all monitored intersections. This is likely the sum of all `[id]_accumulated_waiting_time` columns.
