import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import json # Import the json module
import pandas as pd # Import pandas for DataFrame handling

# --- Start of BatteryDegradationModel from battery_degradation_model_with_compensation.py ---
class BatteryDegradationModel:
    def __init__(self,
                 num_evs: int,
                 ev_initial_soc: float,
                 ev_target_soc: float,
                 ev_battery_capacity_kwh: float,
                 ev_max_charge_rate_kw: float,
                 ev_max_discharge_rate_kw: float, # This parameter is passed but not directly used in current degradation calculations
                 fast_charge_c_rate_threshold: float = 1.0,
                 fast_charge_multiplier: float = 2.0,
                 voltage: float = 360.0):
        self.num_evs = num_evs
        # soc_delta here is based on a single EV's initial/target SOC for the model's internal calc,
        # not directly tied to the LP's per-EV SOCs, but used for internal degradation functions.
        self.soc_delta = abs(ev_target_soc - ev_initial_soc) 
        self.battery_capacity_kwh = ev_battery_capacity_kwh
        self.voltage = voltage
        self.battery_capacity_ah = self.kwh_to_ah(ev_battery_capacity_kwh, voltage)

        self.charge_c_rate = ev_max_charge_rate_kw * 1000 / (self.voltage * self.battery_capacity_ah)
        self.fast_charge_c_rate_threshold = fast_charge_c_rate_threshold
        self.fast_charge_multiplier = fast_charge_multiplier
        self.fast_charge_penalty = self.charge_c_rate > fast_charge_c_rate_threshold

        self.total_ah_throughput_per_ev = 0.0
        self.total_time_days = 0.0
        self.month = 0

        # Degradation Parameters (example values from literature/common models)
        # Calendar aging (time-dependent)
        self.Ea_calendar = 60000.0  # Activation energy for calendar aging (J/mol)
        self.R_gas = 8.314          # Universal gas constant (J/(mol*K))
        self.T_ref_k = 298.15       # Reference temperature for calendar aging (Kelvin, 25°C)
        self.k_cal = 0.001          # Calendar aging coefficient
        self.beta = 0.5             # Calendar aging exponent

        # Cycle aging (throughput-dependent)
        self.k_cyc = 0.0005         # Cycle aging coefficient
        self.alpha = 0.8            # Cycle aging exponent (often < 1 for non-linear degradation)
        self.dod_exponent = 1.0     # Depth of Discharge exponent (simplified, usually more complex)

        self.latest_degradation = {}

    def kwh_to_ah(self, kwh, voltage):
        return (kwh * 1000) / voltage

    def simulate_day(self, daily_ah_throughput_per_ev=None, ambient_temp_celsius=25.0):
        """
        Simulates one day of battery usage and calendar aging.
        :param daily_ah_throughput_per_ev: Average Ah throughput per EV for the day. If None, assumes typical usage.
        :param ambient_temp_celsius: Average ambient temperature for the day in Celsius.
        """
        self.total_time_days += 1
        
        # Simple placeholder for daily_ah_throughput if not provided
        if daily_ah_throughput_per_ev is None:
            # Assume a full cycle equivalent (e.g., 0.6 C-rate discharge for 1 hour)
            # This needs to be more realistically tied to actual charging/discharging from optimizer
            daily_ah_throughput_per_ev = self.battery_capacity_ah * 0.6 * self.soc_delta * 2 # simplified full cycle per day

        self.total_ah_throughput_per_ev += daily_ah_throughput_per_ev

        # Apply fast charging penalty if applicable
        if self.fast_charge_penalty:
            self.total_ah_throughput_per_ev *= (1 + (self.fast_charge_multiplier - 1) * 0.1) # Arbitrary small penalty factor for fast charge

        # Update month counter (simplified, for monthly effects if any)
        if int(self.total_time_days) % 30 == 0:
            self.month += 1

        self.calculate_degradation(ambient_temp_celsius)

    def calculate_degradation(self, ambient_temp_celsius=25.0):
        """
        Calculates the current battery degradation based on accumulated time and throughput.
        """
        T_k = ambient_temp_celsius + 273.15 # Convert Celsius to Kelvin

        # Calendar aging based on time and temperature (Arrhenius)
        calendar_term = np.exp(-self.Ea_calendar / (self.R_gas * T_k))
        cal_loss_frac = self.k_cal * calendar_term * (self.total_time_days**self.beta)

        # Cycle aging based on Ah throughput and DoD (simplified)
        # The (self.soc_delta * self.battery_capacity_ah) term tries to normalize by useful capacity per cycle
        if self.soc_delta > 0: # Avoid division by zero
            cycles_completed = self.total_ah_throughput_per_ev / (self.soc_delta * self.battery_capacity_ah)
        else:
            cycles_completed = 0
            
        cyc_loss_frac = self.k_cyc * (cycles_completed**self.alpha) * (self.soc_delta**self.dod_exponent)

        # Total degradation
        total_loss_frac = cal_loss_frac + cyc_loss_frac # Simple summation

        total_loss_ah_per_ev = total_loss_frac * self.battery_capacity_ah
        total_loss_ah_fleet = total_loss_ah_per_ev * self.num_evs

        self.latest_degradation = {
            "calendar_loss_ah_per_ev": cal_loss_frac * self.battery_capacity_ah,
            "cycle_loss_ah_per_ev": cyc_loss_frac * self.battery_capacity_ah,
            "total_loss_ah_per_ev": total_loss_ah_per_ev,
            "total_loss_percent": 100 * total_loss_frac,
            "total_loss_ah_fleet": total_loss_ah_fleet,
            "fleet_energy_lost_kwh": total_loss_ah_fleet * self.voltage / 1000,
            "fast_charging_applied": self.fast_charge_penalty
        }

        return self.latest_degradation

    def get_remaining_capacity_ah(self):
        if not self.latest_degradation:
            self.calculate_degradation()
        degraded_ah = self.latest_degradation["total_loss_ah_per_ev"]
        return max(0.0, self.battery_capacity_ah - degraded_ah)

    def get_remaining_capacity_kwh(self):
        return self.get_remaining_capacity_ah() * self.voltage / 1000
# --- End of BatteryDegradationModel ---


def v2g_smart_charging_optimizer(
    num_time_steps,
    num_evs,
    grid_base_load,  # Base electricity consumption of the grid (e.g., households, industry)
    grid_predicted_load,  # Predicted additional load on the grid (e.g., from weather, events)
    ev_initial_soc,  # Initial State of Charge for each EV (0-1, as a fraction)
    ev_target_soc,  # Target State of Charge for each EV (0-1, as a fraction)
    ev_battery_capacity,  # Battery capacity of each EV (kWh)
    ev_max_charge_rate,  # Maximum charging rate for each EV (kW)
    ev_max_discharge_rate,  # Maximum discharging rate for each EV (kW)
    electricity_prices,  # Forecasted electricity prices for each time step (€/kWh)
    grid_capacity_limit,  # Maximum power the grid can supply at any time step (kW)
    ev_availability_start_time, # 0-indexed time step when each EV becomes available (plugs in)
    ev_departure_time,    # 0-indexed time step by the end of which each EV must reach target SOC and departs
    # Ancillary Services Parameters
    ancillary_service_up_price,   # Revenue per kW committed for upward regulation (discharge) (€/kW/time_step)
    ancillary_service_down_price, # Revenue per kW committed for downward regulation (charge) (€/kW/time_step)
    min_soc_for_ancillary_discharge, # Minimum SOC (0-1) an EV must maintain to offer upward regulation
    max_soc_for_ancillary_charge,    # Maximum SOC (0-1) an EV can reach and still offer downward regulation
    # General parameters
    # Note: Direct, real-time integration of the provided BatteryDegradationModel
    # into this linear program is not feasible due to its non-linear equations
    # (e.g., power laws, exponentials). This 'battery_degradation_cost_factor'
    # acts as a simplified linear proxy for degradation costs per kWh cycled.
    # Its value could be calibrated using insights from the BatteryDegradationModel
    # through separate simulations or more advanced (non-linear) optimization techniques.
    battery_degradation_cost_factor=0.01, # Cost factor for battery degradation per kWh cycled
    charge_efficiency=0.95,  # Charging efficiency
    discharge_efficiency=0.90,  # Discharging efficiency
    soc_tolerance=0.01, # Tolerance for meeting target SOC
    time_step_duration_hours=1.0 # Duration of each time step in hours (e.g., 1.0 for hourly, 0.25 for 15-min)
):
    """
    Optimizes V2G smart charging/discharging for multiple EVs to minimize electricity cost,
    considering EV availability, departure times, and participation in ancillary services (frequency regulation).
    A linear approximation for battery degradation cost is used.

    Args:
        num_time_steps (int): Number of time steps in the optimization horizon (e.g., 24 for 24 hours).
        num_evs (int): Number of electric vehicles.
        grid_base_load (np.array): Base electricity consumption of the grid for each time step (kW).
        grid_predicted_load (np.array): Predicted additional load on the grid for each time step (kW).
        ev_initial_soc (np.array): Initial State of Charge for each EV (0-1, as a fraction).
        ev_target_soc (np.array): Target State of Charge for each EV (0-1, as a fraction).
        ev_battery_capacity (np.array): Battery capacity of each EV (kWh).
        ev_max_charge_rate (np.array): Maximum charging rate for each EV (kW).
        ev_max_discharge_rate (np.array): Maximum discharging rate for each EV (kW).
        electricity_prices (np.array): Forecasted electricity prices for each time step (€/kWh).
        grid_capacity_limit (float): Maximum power the grid can supply at any time step (kW).
        ev_availability_start_time (np.array): 0-indexed time step when each EV becomes available (plugs in).
        ev_departure_time (np.array): 0-indexed time step by the end of which each EV must
                                       reach target SOC and departs. (e.g., if departs at end of hour 17, this is 17)
        ancillary_service_up_price (float or np.array): Revenue per kW committed for upward regulation (€/kW/time_step).
                                                      Can be a scalar or time-varying array.
        ancillary_service_down_price (float or np.array): Revenue per kW committed for downward regulation (€/kW/time_step).
                                                        Can be a scalar or time-varying array.
        min_soc_for_ancillary_discharge (np.array): Minimum SOC (0-1) an EV must maintain to offer upward regulation.
        max_soc_for_ancillary_charge (np.array): Maximum SOC (0-1) an EV can reach and still offer downward regulation.
        battery_degradation_cost_factor (float): Linear cost factor for battery degradation per kWh cycled.
        charge_efficiency (float): Efficiency of charging (0-1).
        discharge_efficiency (0-1).
        soc_tolerance (float): Allowed deviation from the target SOC at the end.
        time_step_duration_hours (float): Duration of each time step in hours.

    Returns:
        dict: A dictionary containing optimization results:
            - 'charge_power' (np.array): Optimal charging power for each EV at each time step (kW).
            - 'discharge_power' (np.array): Optimal discharging power for each EV at each time step (kW).
            - 'ancillary_up_power' (np.array): Optimal upward regulation power committed by each EV (kW).
            - 'ancillary_down_power' (np.array): Optimal downward regulation power committed by each EV (kW).
            - 'ev_soc' (np.array): State of Charge for each EV at each time step.
            - 'total_cost' (float): Minimum total electricity cost (€).
            - 'grid_total_load' (np.array): Total load on the grid at each time step (kW).
            - 'solver_status' (str): Status of the optimization solver.
    """

    # --- Decision Variables ---
    charge_power = cp.Variable((num_evs, num_time_steps), nonneg=True)
    discharge_power = cp.Variable((num_evs, num_time_steps), nonneg=True)
    ev_soc = cp.Variable((num_evs, num_time_steps + 1))
    
    # Ancillary Service Decision Variables
    ancillary_up_power = cp.Variable((num_evs, num_time_steps), nonneg=True)   # Upward regulation committed power (kW)
    ancillary_down_power = cp.Variable((num_evs, num_time_steps), nonneg=True) # Downward regulation committed power (kW)

    # --- Objective Function ---
    # Minimize total electricity cost minus ancillary service revenue plus degradation cost
    
    # Grid consumption/injection from EVs
    ev_net_power = cp.sum(charge_power - discharge_power, axis=0)
    
    # Total grid load at each time step
    total_grid_load = grid_base_load + grid_predicted_load + ev_net_power

    # Electricity cost (cost of energy arbitrage)
    electricity_cost = cp.sum(cp.multiply(total_grid_load, electricity_prices))

    # Battery Degradation Cost: Proportional to total energy cycled (charged + discharged)
    # This is a linear approximation of degradation.
    total_cycled_energy_kwh = cp.sum(charge_power + discharge_power) * time_step_duration_hours
    degradation_cost = battery_degradation_cost_factor * total_cycled_energy_kwh

    # Ancillary Service Revenue
    ancillary_service_revenue = (
        cp.sum(cp.multiply(ancillary_up_power, ancillary_service_up_price))
        + cp.sum(cp.multiply(ancillary_down_power, ancillary_service_down_price))
    ) * time_step_duration_hours # Multiply by duration for revenue per time step

    # Total objective
    objective = cp.Minimize(electricity_cost + degradation_cost - ancillary_service_revenue)

    # --- Constraints ---
    constraints = []

    # 1. SOC dynamics
    for i in range(num_evs):
        constraints.append(ev_soc[i, 0] == ev_initial_soc[i])  # Initial SOC
        for t in range(num_time_steps):
            constraints.append(
                ev_soc[i, t + 1] == ev_soc[i, t]
                + (charge_power[i, t] * charge_efficiency - discharge_power[i, t] / discharge_efficiency) / ev_battery_capacity[i] * time_step_duration_hours
            )

    # 2. SOC limits (0 to 1)
    constraints.append(ev_soc >= 0)
    constraints.append(ev_soc <= 1)

    # 3. Charging/Discharging limits, EV Availability/Departure, and Ancillary Service Limits
    for i in range(num_evs):
        for t in range(num_time_steps):
            # If EV is not available (not plugged in), no power exchange or service commitment
            if t < ev_availability_start_time[i] or t > ev_departure_time[i]:
                constraints.append(charge_power[i, t] == 0)
                constraints.append(discharge_power[i, t] == 0)
                constraints.append(ancillary_up_power[i, t] == 0)
                constraints.append(ancillary_down_power[i, t] == 0)
                # SOC should remain constant outside the active window IF it's after start, but after departure
                if t >= ev_availability_start_time[i] and t < num_time_steps: # Ensure index is valid for ev_soc[i, t+1]
                    constraints.append(ev_soc[i, t+1] == ev_soc[i, t])
            else:
                # Apply max charge/discharge rates only when available
                # Capacity allocation constraint - physical limits
                # Sum of actual power and committed regulation power cannot exceed max rate
                constraints.append(charge_power[i, t] + ancillary_down_power[i, t] <= ev_max_charge_rate[i])
                constraints.append(discharge_power[i, t] + ancillary_up_power[i, t] <= ev_max_discharge_rate[i])
                
                # An EV cannot truly charge and discharge simultaneously in practice,
                # but in LP, this is typically handled by the optimizer finding the most
                # cost-effective direction. A strict mutual exclusivity requires MILP.
                # The total instantaneous power (sum of charge and discharge)
                # cannot exceed the max of either rate
                constraints.append(charge_power[i, t] + discharge_power[i, t] <= max(ev_max_charge_rate[i], ev_max_discharge_rate[i]))

                # SOC-dependent Ancillary Service Commitment
                # Upward regulation (discharge) requires sufficient available energy
                # The energy equivalent of committed power must be available within SOC bounds
                constraints.append(
                    ancillary_up_power[i, t] * time_step_duration_hours <= (ev_soc[i, t] - min_soc_for_ancillary_discharge[i]) * ev_battery_capacity[i]
                )
                # Downward regulation (charge) requires sufficient empty battery space
                constraints.append(
                    ancillary_down_power[i, t] * time_step_duration_hours <= (max_soc_for_ancillary_charge[i] - ev_soc[i, t]) * ev_battery_capacity[i]
                )
                
                # Ensure committed regulation power doesn't exceed physical limits
                constraints.append(ancillary_up_power[i, t] <= ev_max_discharge_rate[i])
                constraints.append(ancillary_down_power[i, t] <= ev_max_charge_rate[i])


    # 4. Target SOC at the EV's departure time
    for i in range(num_evs):
        # The SOC after the last active time step (ev_departure_time[i])
        soc_at_departure = ev_soc[i, ev_departure_time[i] + 1]
        constraints.append(soc_at_departure >= ev_target_soc[i] * (1 - soc_tolerance))
        constraints.append(soc_at_departure <= ev_target_soc[i] * (1 + soc_tolerance))

    # 5. Grid capacity limit
    constraints.append(total_grid_load <= grid_capacity_limit)
    # Ensure grid load is non-negative
    constraints.append(total_grid_load >= 0)


    # --- Solve the problem ---
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(verbose=True) # ECOS is generally good for LPs
    except Exception as e:
        print(f"Solver error: {e}")
        return {
            'charge_power': None,
            'discharge_power': None,
            'ancillary_up_power': None,
            'ancillary_down_power': None,
            'ev_soc': None,
            'total_cost': None,
            'grid_total_load': None,
            'solver_status': str(e)
        }

    # --- Return Results ---
    if problem.status in ["optimal", "optimal_inaccurate"]:
        return {
            'charge_power': charge_power.value,
            'discharge_power': discharge_power.value,
            'ancillary_up_power': ancillary_up_power.value,
            'ancillary_down_power': ancillary_down_power.value,
            'ev_soc': ev_soc.value,
            'total_cost': problem.value,
            'grid_total_load': total_grid_load.value,
            'solver_status': problem.status
        }
    else:
        print(f"Problem did not solve to optimality. Status: {problem.status}")
        return {
            'charge_power': None,
            'discharge_power': None,
            'ancillary_up_power': None,
            'ancillary_down_power': None,
            'ev_soc': None,
            'total_cost': None,
            'grid_total_load': None,
            'solver_status': problem.status
        }

# --- Function to analyze electricity prices from CSV file ---
def analyze_electricity_prices(file_path='Finland_price_2015_to_2025.csv', target_year=2025):
    """
    Loads electricity price data, calculates the average hourly price profile for a target year,
    and provides examples of how to use this data.

    Args:
        file_path (str): The path to the CSV file containing the price data.
        target_year (int): The year for which to analyze the hourly price profile.

    Returns:
        pandas.DataFrame: A DataFrame containing the average price per hour for the target year
                            in both EUR/MWhe and EUR/KWh.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Convert 'Datetime (Local)' to datetime objects
        df['Datetime (Local)'] = pd.to_datetime(df['Datetime (Local)'])

        # Filter data for the target year
        df_target_year = df[df['Datetime (Local)'].dt.year == target_year].copy()

        if df_target_year.empty:
            print(f"No data found for the year {target_year}.")
            return pd.DataFrame()

        # Extract the hour from the 'Datetime (Local)' column
        df_target_year['Hour'] = df_target_year['Datetime (Local)'].dt.hour

        # Calculate the average price for each hour of the day
        hourly_average_price = df_target_year.groupby('Hour')['Price (EUR/MWhe)'].mean().reset_index()

        # Convert price from EUR/MWhe to EUR/KWh (1 MWh = 1000 KWh)
        hourly_average_price['Price (EUR/KWh)'] = hourly_average_price['Price (EUR/MWhe)'] / 1000

        print(f"\n--- Average Hourly Price Profile for {target_year} ---")
        print(hourly_average_price.round(4))

        # --- Examples of how to "use" this price data ---

        # 1. Get the price for a specific hour (e.g., 10 AM)
        specific_hour = 10
        price_at_10am_KWh = hourly_average_price[hourly_average_price['Hour'] == specific_hour]['Price (EUR/KWh)'].iloc[0]
        print(f"\nAverage price at {specific_hour}:00 for {target_year}: {price_at_10am_KWh:.4f} EUR/KWh")

        # 2. Find the cheapest hour(s)
        cheapest_hour_data = hourly_average_price.loc[hourly_average_price['Price (EUR/KWh)'].idxmin()]
        print(f"\nCheapest hour in {target_year}: Hour {cheapest_hour_data['Hour']} with price {cheapest_hour_data['Price (EUR/KWh)']:.4f} EUR/KWh")

        # 3. Find the most expensive hour(s)
        most_expensive_hour_data = hourly_average_price.loc[hourly_average_price['Price (EUR/KWh)'].idxmax()]
        print(f"Most expensive hour in {target_year}: Hour {most_expensive_hour_data['Hour']} with price {most_expensive_hour_data['Price (EUR/KWh)']:.4f} EUR/KWh")

        # 4. A simple function to get price for a given hour
        def get_price_for_hour(hour, price_profile_df):
            if 0 <= hour <= 23:
                return price_profile_df[price_profile_df['Hour'] == hour]['Price (EUR/KWh)'].iloc[0]
            else:
                return "Invalid hour. Please enter a value between 0 and 23."

        print(f"\nPrice at hour 15: {get_price_for_hour(15, hourly_average_price):.4f} EUR/KWh")

        return hourly_average_price

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    
def calculate_unoptimized_ev_charging_load(
    num_time_steps,
    num_evs,
    ev_initial_soc,
    ev_target_soc,
    ev_battery_capacity,
    ev_max_charge_rate,
    ev_availability_start_time,
    ev_departure_time,
    charge_efficiency,
    time_step_duration_hours
):
    """
    Calculates the unoptimized (greedy) charging load for all EVs.
    Each EV charges at its max rate whenever available until it reaches target SOC.
    """
    unoptimized_ev_charging_power = np.zeros((num_evs, num_time_steps))
    current_soc = np.copy(ev_initial_soc)

    for t in range(num_time_steps):
        for i in range(num_evs):
            # Check if EV is available in the current time step
            if ev_availability_start_time[i] <= t <= ev_departure_time[i]:
                # Calculate needed charge to reach target SOC
                needed_charge_kwh = (ev_target_soc[i] - current_soc[i]) * ev_battery_capacity[i]
                
                # If battery needs charging
                if needed_charge_kwh > 1e-6: # Using a small tolerance for float comparison
                    # Calculate how much power can be drawn in this time step
                    max_charge_this_step_kwh = ev_max_charge_rate[i] * time_step_duration_hours * charge_efficiency
                    
                    # Charge up to what's needed or max rate allows
                    actual_charge_kwh = min(needed_charge_kwh, max_charge_this_step_kwh)
                    
                    # Convert energy (kWh) back to power (kW) for this time step
                    unoptimized_ev_charging_power[i, t] = actual_charge_kwh / (time_step_duration_hours * charge_efficiency)
                    
                    # Update SOC
                    current_soc[i] += actual_charge_kwh / ev_battery_capacity[i]
                    current_soc[i] = np.clip(current_soc[i], 0.0, 1.0) # Ensure SOC stays within bounds
                else:
                    unoptimized_ev_charging_power[i, t] = 0 # Already at or above target SOC
            else:
                unoptimized_ev_charging_power[i, t] = 0 # EV not available

    return np.sum(unoptimized_ev_charging_power, axis=0) # Sum across all EVs for total load per time step

# --- Main execution block ---
if __name__ == "__main__":
    # Define parameters for a 24-hour simulation
    num_time_steps = 24  # Hourly steps (0 to 23)
    time_step_duration_hours = 1.0 # Each time step is 1 hour
    charge_efficiency_for_check = 0.95 # Define here for use in feasibility check

    # Time-varying grid data (example - replace with real data)
    grid_base_load = np.array([500, 480, 470, 460, 470, 500, 550, 600, 620, 600, 580, 570,
                               560, 550, 540, 530, 550, 600, 650, 700, 680, 650, 600, 550]) # kW
    grid_predicted_load = np.array([50, 40, 30, 20, 30, 50, 80, 100, 120, 100, 90, 80,
                                    70, 60, 50, 40, 60, 100, 150, 200, 180, 150, 100, 70]) # kW (e.g., HVAC load)

    # --- Use the new function to get electricity prices ---
    # NOTE: This function expects 'Finland_price_2015_to_2025.csv' to be available.
    # If not present, it will print an error and return an empty DataFrame,
    # which will then cause an issue when converting to numpy array.
    # Therefore, we will assign a default numpy array to electricity_prices
    # if the function returns an empty DataFrame.
    hourly_prices_df = analyze_electricity_prices(file_path='Finland_price_2015_to_2025.csv', target_year=2025)
    if not hourly_prices_df.empty:
        electricity_prices = hourly_prices_df['Price (EUR/KWh)'].values
        # Ensure electricity_prices array has the correct number of time steps
        if len(electricity_prices) < num_time_steps:
            print(f"Warning: Loaded prices ({len(electricity_prices)}) fewer than num_time_steps ({num_time_steps}). Padding with last value.")
            electricity_prices = np.pad(electricity_prices, (0, num_time_steps - len(electricity_prices)), 'edge')
        elif len(electricity_prices) > num_time_steps:
            electricity_prices = electricity_prices[:num_time_steps]
            print(f"Warning: Loaded prices ({len(electricity_prices)}) more than num_time_steps ({num_time_steps}). Truncating.")
    else:
        print("Using default electricity prices as CSV data could not be loaded.")
        electricity_prices = np.array([0.15, 0.14, 0.13, 0.12, 0.13, 0.16, 0.18, 0.22, 0.25, 0.23, 0.20, 0.18,
                                       0.17, 0.16, 0.15, 0.14, 0.18, 0.23, 0.28, 0.30, 0.28, 0.25, 0.20, 0.16])


    grid_capacity_limit = 1500.0  # kW (example: total capacity of a local transformer or substation)

    # --- Read EV parameters from EV_info.txt ---
    print("\n--- Reading EV Parameters from EV_info.txt ---")
    try:
        ev_info_data = np.loadtxt('EV_info.txt')
        if ev_info_data.ndim == 1: # Handle case of a single EV (loadtxt returns 1D array)
            ev_info_data = ev_info_data.reshape(1, -1)
        
        num_evs = ev_info_data.shape[0]

        # Columns from EV_info.txt (assuming 1-indexed hours, convert to 0-indexed time steps)
        # EV_info(:,1)=T_arrival;
        ev_availability_start_time = (ev_info_data[:, 0] - 1).astype(int)
        # EV_info(:,2)=T_departure;
        ev_departure_time = (ev_info_data[:, 1] - 1).astype(int)
        # EV_info(:,3)=Cap_battery_org*Ini_percentage; (Initial energy in kWh)
        initial_energy_kwh = ev_info_data[:, 2]

        print(f"Detected {num_evs} EVs from EV_info.txt.")
        print(f"Arrival times (0-indexed): {ev_availability_start_time}")
        print(f"Departure times (0-indexed): {ev_departure_time}")
        print(f"Initial energy (kWh): {initial_energy_kwh}")

        # Default EV parameters - these will be extended/truncated based on num_evs from file
        # We assume a base EV's characteristics and apply them to all EVs from the file
        # unless specific columns for these exist in EV_info.txt (which they don't in the snippet).
        base_ev_capacity = 30.0 # kWh
        base_max_charge_rate = 11.0 # kW
        base_max_discharge_rate = 11.0 # kW
        base_target_soc = 0.8
        base_min_soc_ancillary_discharge = 0.0
        base_max_soc_ancillary_charge = 0.9

        # Create arrays for EV parameters, replicating base values for new EVs
        ev_battery_capacity = np.full(num_evs, base_ev_capacity)
        ev_max_charge_rate = np.full(num_evs, base_max_charge_rate)
        ev_max_discharge_rate = np.full(num_evs, base_max_discharge_rate)
        ev_target_soc = np.full(num_evs, base_target_soc)
        min_soc_for_ancillary_discharge = np.full(num_evs, base_min_soc_ancillary_discharge)
        max_soc_for_ancillary_charge = np.full(num_evs, base_max_soc_ancillary_charge)

        # Calculate initial SOC for each EV using its initial energy and assumed battery capacity
        # Handle division by zero if capacity is zero for some reason
        ev_initial_soc = np.divide(initial_energy_kwh, ev_battery_capacity, 
                                   out=np.zeros_like(initial_energy_kwh, dtype=float), 
                                   where=ev_battery_capacity!=0)
        # Clamp SOC values to be between 0 and 1
        ev_initial_soc = np.clip(ev_initial_soc, 0.0, 1.0)


    except FileNotFoundError:
        print("Error: EV_info.txt not found. Using default example EV parameters.")
        num_evs = 10
        ev_initial_soc = np.array([0.5, 0.3, 0.4])
        ev_target_soc = np.array([0.9, 0.8, 0.9])
        ev_battery_capacity = np.array([60.0, 80.0, 40.0])
        ev_max_charge_rate = np.array([11.0, 22.0, 7.0])
        ev_max_discharge_rate = np.array([11.0, 15.0, 5.0])
        ev_availability_start_time = np.array([7, 16, 8])
        ev_departure_time = np.array([17, 23, 16])
        min_soc_for_ancillary_discharge = np.array([0.2, 0.25, 0.15])
        max_soc_for_ancillary_charge = np.array([0.95, 0.9, 0.98])
    except Exception as e:
        print(f"Error reading EV_info.txt: {e}. Using default example EV parameters.")
        num_evs = 3
        ev_initial_soc = np.array([0.5, 0.3, 0.4])
        ev_target_soc = np.array([0.9, 0.8, 0.9])
        ev_battery_capacity = np.array([60.0, 80.0, 40.0])
        ev_max_charge_rate = np.array([11.0, 22.0, 7.0])
        ev_max_discharge_rate = np.array([11.0, 15.0, 5.0])
        ev_availability_start_time = np.array([0, 8, 17])
        ev_departure_time = np.array([7, 16, 23])
        min_soc_for_ancillary_discharge = np.array([0.2, 0.25, 0.15])
        max_soc_for_ancillary_charge = np.array([0.95, 0.9, 0.98])


    # --- Feasibility Check for EVs ---
    print("\n--- Performing Feasibility Check for EV Charging Schedules ---")
    infeasible_ev_count = 0
    for i in range(num_evs):
        required_charge_kwh = max(0, (ev_target_soc[i] - ev_initial_soc[i]) * ev_battery_capacity[i])
        available_time_hours = ev_departure_time[i] - ev_availability_start_time[i] + 1 # +1 for inclusive hours

        # Ensure that max_charge_rate is not zero to avoid division by zero
        if ev_max_charge_rate[i] > 0:
            min_time_needed_hours = required_charge_kwh / (ev_max_charge_rate[i] * charge_efficiency_for_check)
        else:
            min_time_needed_hours = float('inf') # Cannot charge if rate is 0

        # Check for immediate charging period validity 
        if available_time_hours <= 0:
            status_msg = f"EV {i+1}: Invalid or zero charging period ({available_time_hours} hours). Arrival: {ev_availability_start_time[i]}, Departure: {ev_departure_time[i]}."
            is_feasible = False
        elif min_time_needed_hours > available_time_hours:
            status_msg = (f"EV {i+1}: Cannot reach target SOC. Needs {min_time_needed_hours:.2f} hrs, "
                          f"but only {available_time_hours} hrs available. "
                          f"(Initial SOC: {ev_initial_soc[i]:.2f}, Target SOC: {ev_target_soc[i]:.2f}, "
                          f"Capacity: {ev_battery_capacity[i]:.1f} kWh, Max Charge Rate: {ev_max_charge_rate[i]:.1f} kW)")
            is_feasible = False
        else:
            status_msg = f"EV {i+1}: Schedule appears feasible (Needs {min_time_needed_hours:.2f} hrs, Has {available_time_hours} hrs)."
            is_feasible = True

        if not is_feasible:
            infeasible_ev_count += 1

        if i < 5: # Print details for the first 5 EVs
            print(status_msg)

    if num_evs > 5:
        print(f"\nSummary for remaining {num_evs - 5} EVs: {infeasible_ev_count} out of {num_evs} EVs have potentially infeasible schedules.")
    elif num_evs > 0:
        print(f"\nOverall: {infeasible_ev_count} out of {num_evs} EVs have potentially infeasible schedules.")
    else:
        print("No EVs loaded for feasibility check.")

    # Ancillary Services Parameters (can be scalars or arrays matching num_time_steps)
    ancillary_service_up_price = 0.05   # Example: €0.05/kW/hour for discharge regulation
    ancillary_service_down_price = 0.04 # Example: €0.04/kW/hour for charge regulation

    # Additional debug prints for ancillary service SOC bounds
    print("\n--- Ancillary Service SOC Bounds Check ---")
    for i in range(num_evs):
        print(f"EV {i+1}: Min SOC for Discharge: {min_soc_for_ancillary_discharge[i]:.2f}, Max SOC for Charge: {max_soc_for_ancillary_charge[i]:.2f}")
        if min_soc_for_ancillary_discharge[i] >= max_soc_for_ancillary_charge[i]:
            print(f"WARNING: EV {i+1} has an invalid or conflicting ancillary service SOC range (Min >= Max). This may cause infeasibility or unboundedness.")
    
    # --- Battery Degradation Model Calibration ---
    print("\n--- Calibrating Battery Degradation Cost Factor ---")
    # Define parameters for a single 'representative' EV for calibration purposes
    # This calibration assumes all EVs have roughly the same degradation characteristics.
    cal_ev_capacity_kwh = ev_battery_capacity[0] if num_evs > 0 else 60.0
    cal_ev_voltage = 360.0 # Assumed voltage for calibration
    
    cal_model = BatteryDegradationModel(
        num_evs=1, 
        ev_initial_soc=0.2, 
        ev_target_soc=0.8,
        ev_battery_capacity_kwh=cal_ev_capacity_kwh,
        ev_max_charge_rate_kw=ev_max_charge_rate[0] if num_evs > 0 else 11.0,
        ev_max_discharge_rate_kw=ev_max_discharge_rate[0] if num_evs > 0 else 11.0,
        voltage=cal_ev_voltage
    )

    calibration_days = 365 # Simulate for one year
    # Assume an average daily Ah throughput for cycling.
    # A full cycle (battery_capacity_ah) is a common reference.
    daily_ah_throughput_for_calibration = cal_model.battery_capacity_ah 

    total_ah_throughput_accumulated = 0.0

    for day in range(calibration_days):
        cal_model.simulate_day(daily_ah_throughput_per_ev=daily_ah_throughput_for_calibration)
        total_ah_throughput_accumulated += daily_ah_throughput_for_calibration
        
    final_degradation_result = cal_model.latest_degradation
    total_lost_capacity_ah = final_degradation_result["total_loss_ah_per_ev"]
    total_lost_capacity_kwh = total_lost_capacity_ah * cal_ev_voltage / 1000

    total_throughput_kwh_for_calibration = total_ah_throughput_accumulated * cal_ev_voltage / 1000

    # Hypothetical cost of replacing lost capacity (e.g., $200 per kWh of lost capacity)
    cost_per_kwh_replacement = 200.0 # €/kWh of lost capacity
    total_degradation_cost_in_calibration = total_lost_capacity_kwh * cost_per_kwh_replacement

    # Calibrated degradation cost factor (€/kWh cycled)
    # This is the total cost incurred due to degradation divided by the total energy cycled.
    if total_throughput_kwh_for_calibration > 0:
        calibrated_degradation_factor = total_degradation_cost_in_calibration / total_throughput_kwh_for_calibration
    else:
        calibrated_degradation_factor = 0.0 # No throughput, no degradation cost

    print(f"Calibration for {calibration_days} days for a {cal_ev_capacity_kwh} kWh EV at 1 equivalent full cycle/day:")
    print(f"Total throughput during calibration: {total_throughput_kwh_for_calibration:.2f} kWh")
    print(f"Total capacity lost: {total_lost_capacity_kwh:.2f} kWh ({final_degradation_result['total_loss_percent']:.2f}%)")
    print(f"Estimated degradation cost in calibration: €{total_degradation_cost_in_calibration:.2f}")
    print(f"Calibrated battery_degradation_cost_factor: €{calibrated_degradation_factor:.4f} per kWh cycled")

    # Use the calibrated factor in the optimizer
    linear_battery_degradation_cost_per_kwh_cycled = calibrated_degradation_factor
    # --- End of Battery Degradation Model Calibration ---
    
# --- Calculate Unoptimized EV Charging Load ---
    print("\n--- Calculating Unoptimized (Greedy) EV Charging Load ---")
    unoptimized_ev_load = calculate_unoptimized_ev_charging_load(
        num_time_steps,
        num_evs,
        ev_initial_soc,
        ev_target_soc,
        ev_battery_capacity,
        ev_max_charge_rate,
        ev_availability_start_time,
        ev_departure_time,
        charge_efficiency_for_check,
        time_step_duration_hours
    )
    unoptimized_total_grid_load = grid_base_load + grid_predicted_load + unoptimized_ev_load

    print("Running V2G Smart Charging Optimizer with Ancillary Services and Calibrated Battery Degradation...")
    results = v2g_smart_charging_optimizer(
        num_time_steps,
        num_evs,
        grid_base_load,
        grid_predicted_load,
        ev_initial_soc,
        ev_target_soc,
        ev_battery_capacity,
        ev_max_charge_rate,
        ev_max_discharge_rate,
        electricity_prices,
        grid_capacity_limit,
        ev_availability_start_time,
        ev_departure_time,
        ancillary_service_up_price,
        ancillary_service_down_price,
        min_soc_for_ancillary_discharge,
        max_soc_for_ancillary_charge,
        battery_degradation_cost_factor=linear_battery_degradation_cost_per_kwh_cycled,
        charge_efficiency=charge_efficiency_for_check, # Pass defined charge_efficiency
        time_step_duration_hours=time_step_duration_hours
    )

    if results['solver_status'] in ["optimal", "optimal_inaccurate"]:
        print("\n--- Optimization Results ---")
        print(f"Total Optimized Cost (including ancillary service revenue): €{results['total_cost']:.2f}")
        print(f"Solver Status: {results['solver_status']}")

        """ print("\nOptimal Charging Power (kW per EV, per hour):")
        for i in range(num_evs):
            print(f"EV {i+1}: {np.round(results['charge_power'][i], 2)}")

        print("\nOptimal Discharging Power (kW per EV, per hour):")
        for i in range(num_evs):
            print(f"EV {i+1}: {np.round(results['discharge_power'][i], 2)}")

        print("\nOptimal Upward Regulation Power Committed (kW per EV, per hour):")
        for i in range(num_evs):
            print(f"EV {i+1}: {np.round(results['ancillary_up_power'][i], 2)}")

        print("\nOptimal Downward Regulation Power Committed (kW per EV, per hour):")
        for i in range(num_evs):
            print(f"EV {i+1}: {np.round(results['ancillary_down_power'][i], 2)}")

        print("\nEV State of Charge (SOC, 0-1) over time:")
        for i in range(num_evs):
            print(f"EV {i+1}: {np.round(results['ev_soc'][i], 2)}")

        print("\nTotal Grid Load (kW) including EVs:")
        print(np.round(results['grid_total_load'], 2)) """

        # --- Optimization Plots ---
        try:
                        # --- Figure 1: Electricity Prices and Grid Load Profile ---
            # This plot combines grid load information with electricity prices on a secondary Y-axis.
            fig1, ax1 = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability

            # Plot Grid Load Profile on the primary Y-axis (ax1)
            ax1.plot(np.arange(num_time_steps), grid_base_load + grid_predicted_load, label='Grid Base + Predicted Load', color='blue')
            plt.plot(unoptimized_total_grid_load, label='Unoptimized Total Grid Load (Greedy EV)', linestyle='--') # Added unoptimized load

            ax1.plot(np.arange(num_time_steps), results['grid_total_load'], label='Grid Total Load (with EVs)', color='green')
            ax1.axhline(y=grid_capacity_limit, color='red', linestyle='--', label='Grid Capacity Limit')
            ax1.set_xlabel('Time Step (Hour)')
            ax1.set_ylabel('Power (kW)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('Electricity Prices and Grid Load Profile')
            ax1.grid(True)

            # Create a second Y-axis (twinx) for Electricity Prices
            ax2 = ax1.twinx()
            ax2.plot(np.arange(num_time_steps), electricity_prices, label='Electricity Price (€/kWh)', color='purple', linestyle=':')
            ax2.set_ylabel('Price (€/kWh)', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1)) # Place legend outside

            fig1.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legend
            plt.savefig('electricity_prices_and_grid_load_profile.png') # Save the plot
            plt.close(fig1) # Close the figure to free up memory

            # --- Figure 2: EV Charging/Discharging Power and Ancillary Service Power ---
            # This figure contains two subplots displaying EV power flows.
            fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Share X-axis for alignment

            # Plot 3: EV Charging/Discharging Power (Positive: Charge, Negative: Discharge)
            for i in range(num_evs):
                ax3.plot(np.arange(num_time_steps), results['charge_power'][i], label=f'EV {i+1} Charge Power', alpha=0.7)
                ax3.plot(np.arange(num_time_steps), -results['discharge_power'][i], label=f'EV {i+1} Discharge Power', linestyle='--', alpha=0.7)
            ax3.set_title('EV Charging/Discharging Power')
            ax3.set_ylabel('Power (kW)')
            ax3.grid(True)
            ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1)) # Arrange legend in two columns for space

            # Plot 4: EV Ancillary Service Power Committed (kW)
            for i in range(num_evs):
                ax4.plot(np.arange(num_time_steps), results['ancillary_up_power'][i], label=f'EV {i+1} Ancillary Up Power', alpha=0.7)
                ax4.plot(np.arange(num_time_steps), results['ancillary_down_power'][i], label=f'EV {i+1} Ancillary Down Power', linestyle=':', alpha=0.7)
            ax4.set_title('EV Ancillary Service Power Committed')
            ax4.set_xlabel('Time Step (Hour)')
            ax4.set_ylabel('Power (kW)')
            ax4.grid(True)
            ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

            fig2.tight_layout()
            plt.savefig('ev_charging_and_ancillary_power.png') # Save the plot
            #plt.close(fig2) # Close the figure

            # --- Figure 3: EV State of Charge ---
            # This plot shows the evolution of each EV's State of Charge.
            fig3, ax5 = plt.subplots(figsize=(12, 7)) # Separate figure for SOC

            for i in range(num_evs):
                # Get a distinct color for each EV's line
                color = plt.cm.viridis(i / num_evs) 
                ax5.plot(np.arange(num_time_steps + 1), results['ev_soc'][i], label=f'EV {i+1} SOC', color=color, linewidth=2)
                
                # Add markers for start and departure times
                ax5.axvline(x=ev_availability_start_time[i], color=color, linestyle=':', label=f'EV {i+1} Avail Start')
                ax5.axvline(x=ev_departure_time[i], color=color, linestyle='-.', label=f'EV {i+1} Depart Time')
                
                # Add target SOC line
                ax5.axhline(y=ev_target_soc[i], linestyle='--', color=color, label=f'EV {i+1} Target SOC')

            ax5.set_title('EV State of Charge')
            ax5.set_xlabel('Time Step (Hour)')
            ax5.set_ylabel('SOC (0-1)')
            ax5.set_ylim(0, 1) # SOC is typically between 0 and 1
            ax5.grid(True)
            ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1) # Place legend outside for clarity

            fig3.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
            plt.savefig('ev_state_of_charge.png') # Save the plot
            #plt.close(fig3) # Close the figure

            #print("Plots have been generated and saved as:")
            #print("- electricity_prices_and_grid_load_profile.png")
            #print("- ev_charging_and_ancillary_power.png")
            #print("- ev_state_of_charge.png")
           
        except ImportError:
            print("\nMatplotlib not installed. Skipping optimization plot generation.")
            print("To install: pip install matplotlib")

        # --- Illustrative Battery Degradation Model Usage (External to LP) for ALL EVs ---
        print("\n--- Illustrative Battery Degradation Model Usage (External to LP) for ALL EVs ---")
        
        all_ev_cumulative_loss_percent = []
        days_to_simulate_analysis = 30 # Number of days to simulate degradation for each EV

        for i in range(num_evs):
            #print(f"\nSimulating degradation for EV {i+1} based on its throughput from LP results:")
            # Use the actual battery capacity, max charge/discharge rates of the current EV
            ev_model_analysis = BatteryDegradationModel(
                num_evs=1, 
                ev_initial_soc=ev_initial_soc[i],
                ev_target_soc=ev_target_soc[i],
                ev_battery_capacity_kwh=ev_battery_capacity[i],
                ev_max_charge_rate_kw=ev_max_charge_rate[i],
                ev_max_discharge_rate_kw=ev_max_discharge_rate[i],
                voltage=cal_ev_voltage # Use the same calibration voltage for consistency
            )

            # Calculate total energy throughput (charge + discharge) from the LP results for this EV
            total_charge_kwh_ev = np.sum(results['charge_power'][i]) * time_step_duration_hours
            total_discharge_kwh_ev = np.sum(results['discharge_power'][i]) * time_step_duration_hours
            
            # Approximate daily average throughput for analysis from the total over the optimization horizon
            # This assumes the daily cycling behavior is consistent with the average of the 24-hour LP output
            if ev_model_analysis.voltage > 0 and ev_model_analysis.battery_capacity_ah > 0: # Avoid division by zero
                daily_ah_throughput_ev_analysis = (total_charge_kwh_ev + total_discharge_kwh_ev) * 1000 / ev_model_analysis.voltage / num_time_steps
            else:
                daily_ah_throughput_ev_analysis = 0.0

            #print(f"Approximate Daily AH Throughput for EV {i+1}: {daily_ah_throughput_ev_analysis:.2f} Ah/day")
            
            cumulative_total_loss_percent_ev = []
            for day in range(days_to_simulate_analysis):
                ev_model_analysis.simulate_day(daily_ah_throughput_per_ev=daily_ah_throughput_ev_analysis)
                degradation_result_analysis = ev_model_analysis.calculate_degradation()
                cumulative_total_loss_percent_ev.append(degradation_result_analysis["total_loss_percent"])
            
            all_ev_cumulative_loss_percent.append(cumulative_total_loss_percent_ev)

            #print(f"After {days_to_simulate_analysis} days, EV {i+1} total capacity loss: {degradation_result_analysis['total_loss_percent']:.2f}%")
            #print(f"Corresponding energy lost: {degradation_result_analysis['fleet_energy_lost_kwh']:.2f} kWh (for EV {i+1})")

        # Plot cumulative degradation for ALL EVs
        plt.figure(figsize=(10, 6))
        for i, losses in enumerate(all_ev_cumulative_loss_percent):
            plt.plot(np.arange(1, days_to_simulate_analysis + 1), losses, label=f'EV {i+1} Degradation (%)')
        plt.title('Illustrative Battery Degradation Over Time for All EVs (from LP-derived throughput)')
        plt.xlabel('Days')
        plt.ylabel('Capacity Loss (%)')
        plt.grid(True)
        plt.legend()
        plt.show()
        #plt.savefig('all_ev_cumulative_loss_percent.png') # Save the plot

    else:
        print(f"\nOptimization failed. Solver status: {results['solver_status']}")
        print("Check constraints and input parameters.")
