import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters & Distributions ---
num_evs = 1000 # Number of EVs in our simulated fleet

# Home Charging Parameters
mu_distance = 40; sigma_distance = 15
mu_efficiency = 0.18; sigma_efficiency = 0.02
home_arrival_time_mu = 18.0; home_arrival_time_sigma = 2.5 # Home arrival
home_departure_time_mu = 7.0; home_departure_time_sigma = 1.5 # Home departure

home_charging_powers = [3.7, 7.4, 11] # kW (typical Level 2)
home_charging_probabilities = [0.2, 0.6, 0.2]

# Battery Capacities (fixed per EV for consistency between runs)
# These will be generated once at the start of the simulation setup
# and then passed to the simulation functions.

desired_soc_at_departure_range = [0.8, 1.0] # 80-100%

# Public Charging Parameters
public_charging_prob = 0.2 # 20% chance of a public charging event

# GMM for Public Charging Arrival Times (from provided table)
gmm_components_data = [
    {'p': 0.27, 'mu': 9.2,  'sigma': 0.99},
    {'p': 0.03, 'mu': 10.30, 'sigma': 0.44},
    {'p': 0.00, 'mu': 14.05, 'sigma': 0.18}, # Component with 0 probability
    {'p': 0.33, 'mu': 16.73, 'sigma': 1.64},
    {'p': 0.14, 'mu': 11.93, 'sigma': 0.70},
    {'p': 0.02, 'mu': 15.95, 'sigma': 0.43},
    {'p': 0.04, 'mu': 15.05, 'sigma': 0.64},
    {'p': 0.18, 'mu': 13.66, 'sigma': 0.91}
]

active_gmm_weights = [c['p'] for c in gmm_components_data if c['p'] > 0]
active_gmm_mus = [c['mu'] for c in gmm_components_data if c['p'] > 0]
active_gmm_sigmas = [c['sigma'] for c in gmm_components_data if c['p'] > 0]
sum_weights = sum(active_gmm_weights)
normalized_gmm_weights = [w / sum_weights for w in active_gmm_weights]

# Public Charging Duration and Power
public_duration_options = [0.5, 1, 1.5, 2, 2.5] # hours
public_duration_probs = [0.15, 0.4, 0.25, 0.1, 0.1]
public_charger_powers = [22, 50, 150] # kW (Level 2 public, DCFC)
public_charger_probs = [0.3, 0.5, 0.2]

# Grid Parameters for Smart Charging
hourly_prices = np.array([ # Example dynamic electricity prices (24 hours)
    0.10, 0.09, 0.09, 0.08, 0.08, 0.09, # 0-5 AM (off-peak)
    0.12, 0.15, 0.18, 0.20, 0.22, 0.20, # 6-11 AM (rising, peak)
    0.18, 0.16, 0.15, 0.14, 0.16, 0.25, # 12-5 PM (mid-day, evening peak starts)
    0.30, 0.28, 0.25, 0.20, 0.15, 0.12  # 6-11 PM (peak, falling)
])

home_grid_limit_kw = 20 # kW, Example total household limit (e.g., main fuse capacity)
public_avg_price_per_kwh = 0.35 # $/kWh (example average public charging price)

# --- Cost Calculation Function ---
def calculate_charging_cost(hourly_demand_profile_kw, hourly_prices_per_kwh):
    """
    Calculates the total cost of charging for a given hourly demand profile.
    Assumes demand is in kW and prices are in $/kWh.
    """
    # Demand profile is kW, and each entry represents 1 hour's worth of power usage.
    # So, kW * 1 hour = kWh. Summing kW values directly gives total kWh for the day.
    # This also means the price array needs to align hourly.
    
    # Ensure arrays are of the same length
    if len(hourly_demand_profile_kw) != len(hourly_prices_per_kwh):
        raise ValueError("Hourly demand profile and hourly prices must have the same length (24 hours).")

    # Calculate energy consumed per hour (kW * 1h = kWh) and then multiply by price
    hourly_energy_kwh = hourly_demand_profile_kw * 1 # Since our demand is already kW over an hour slot
    hourly_costs = hourly_energy_kwh * hourly_prices_per_kwh
    
    total_cost = np.sum(hourly_costs)
    return total_cost


# Function to sample from the GMM
def sample_from_gmm(weights, mus, sigmas):
    chosen_component_idx = np.random.choice(len(weights), p=weights)
    sample = np.random.normal(mus[chosen_component_idx], sigmas[chosen_component_idx])
    return np.clip(sample, 0, 23.99)

# --- Generate Base EV Parameters (Run once) ---
np.random.seed(42) # for reproducibility

base_ev_params = []
for i in range(num_evs):
    base_ev_params.append({
        'EV_ID': i,
        'Daily_Distance_km': max(0, np.random.normal(mu_distance, sigma_distance)),
        'Vehicle_Efficiency_kWh_km': max(0.1, np.random.normal(mu_efficiency, sigma_efficiency)),
        'Battery_Capacity_kWh': np.random.uniform(40, 80), #np.random.uniform(battery_capacities.min(), battery_capacities.max()), # Sample from the defined range
        'Initial_SoC': np.random.uniform(0.5, 1.0), # Start of day SoC
        'Desired_SoC_at_Departure': np.random.uniform(desired_soc_at_departure_range[0], desired_soc_at_departure_range[1]),
        'Home_Arrival_Time_h': np.clip(np.random.normal(home_arrival_time_mu, home_arrival_time_sigma), 0, 23.99),
        'Home_Departure_Time_h': np.clip(np.random.normal(home_departure_time_mu, home_departure_time_sigma), 0, 23.99),
        'Home_Charging_Power_kW': np.random.choice(home_charging_powers, p=home_charging_probabilities),
        'Public_Charging_Decision': np.random.rand() < public_charging_prob # True/False if they consider public charging
    })

# --- Simulation Functions ---

def simulate_charging(ev_params_list, simulation_type='smart'):
    """
    Simulates EV charging for a list of EV parameters, either unoptimized or smart.
    Returns individual EV records and aggregated hourly demand.
    """
    ev_records = []
    total_hourly_demand = np.zeros(24)
    total_hourly_public_demand = np.zeros(24) # Public demand is the same for both scenarios

    for ev_param in ev_params_list:
        ev_id = ev_param['EV_ID']
        daily_distance = ev_param['Daily_Distance_km']
        vehicle_efficiency = ev_param['Vehicle_Efficiency_kWh_km']
        battery_capacity = ev_param['Battery_Capacity_kWh']
        initial_soc = ev_param['Initial_SoC']
        desired_soc_at_departure = ev_param['Desired_SoC_at_Departure']
        home_arrival_time = ev_param['Home_Arrival_Time_h']
        home_departure_time = ev_param['Home_Departure_Time_h']
        home_charger_power = ev_param['Home_Charging_Power_kW']
        public_charging_decision = ev_param['Public_Charging_Decision']

        current_soc_kwh = initial_soc * battery_capacity
        desired_soc_kwh = desired_soc_at_departure * battery_capacity

        # Energy consumed during travel
        energy_consumed_travel_kwh = daily_distance * vehicle_efficiency
        current_soc_kwh -= energy_consumed_travel_kwh

        # Public Charging Event? (Common to both smart and unoptimized)
        public_charged_kwh = 0
        public_arrival_hour = np.nan # Default to NaN if no public charge

        if public_charging_decision and current_soc_kwh < desired_soc_kwh:
            public_charger_kw = np.random.choice(public_charger_powers, p=public_charger_probs)
            public_duration_hours = np.random.choice(public_duration_options, p=public_duration_probs)
            public_arrival_hour = sample_from_gmm(normalized_gmm_weights, active_gmm_mus, active_gmm_sigmas)

            public_charged_kwh = min(public_duration_hours * public_charger_kw, desired_soc_kwh - current_soc_kwh)
            public_charged_kwh = max(0, public_charged_kwh)
            current_soc_kwh += public_charged_kwh

            for h in range(int(public_arrival_hour), min(24, int(public_arrival_hour + public_duration_hours))):
                total_hourly_public_demand[h] += public_charger_kw

        # Calculate Home Charging Energy Gap
        energy_gap_kwh = desired_soc_kwh - current_soc_kwh
        energy_gap_kwh = max(0, energy_gap_kwh)

        home_charging_schedule = np.zeros(24)

        if energy_gap_kwh > 0:
            if simulation_type == 'unoptimized':
                unoptimized_remaining_energy = energy_gap_kwh
                unoptimized_current_hour = int(home_arrival_time)
                
                # Maximum hours available for charging
                hours_available = home_departure_time - home_arrival_time
                if hours_available < 0: # Overnight case
                    hours_available += 24

                hours_passed = 0
                while unoptimized_remaining_energy > 0 and hours_passed < int(hours_available) + 1: # Loop through available full hours
                    hour_idx = unoptimized_current_hour % 24
                    
                    # Power that can be drawn in this hour
                    power_to_draw_this_hour = min(home_charger_power, unoptimized_remaining_energy)
                    
                    home_charging_schedule[hour_idx] += power_to_draw_this_hour
                    unoptimized_remaining_energy -= power_to_draw_this_hour
                    
                    unoptimized_current_hour = (unoptimized_current_hour + 1) % 24
                    hours_passed += 1

            elif simulation_type == 'smart':
                smart_remaining_energy = energy_gap_kwh
                smart_arrival_hour_int = int(home_arrival_time)
                smart_departure_hour_int = int(home_departure_time)

                smart_available_hours_indices = []
                if smart_departure_hour_int > smart_arrival_hour_int:
                    smart_available_hours_indices = list(range(smart_arrival_hour_int, smart_departure_hour_int))
                else:
                    smart_available_hours_indices = list(range(smart_arrival_hour_int, 24)) + list(range(0, smart_departure_hour_int))

                charge_priority_smart = sorted(smart_available_hours_indices, key=lambda h: hourly_prices[h])

                for hour_idx in charge_priority_smart:
                    if smart_remaining_energy <= 0:
                        break
                    power_draw_smart = min(home_charger_power, smart_remaining_energy)
                    home_charging_schedule[hour_idx] = power_draw_smart
                    smart_remaining_energy -= power_draw_smart
            else:
                raise ValueError("simulation_type must be 'unoptimized' or 'smart'")

        # Store results for this EV
        record = {
            'EV_ID': ev_id,
            'Daily_Distance_km': daily_distance,
            'Home_Arrival_Time_h': home_arrival_time,
            'Home_Departure_Time_h': home_departure_time,
            'Battery_Capacity_kWh': battery_capacity,
            'Initial_SoC_kWh': initial_soc * battery_capacity,
            'Energy_Consumed_Travel_kWh': energy_consumed_travel_kwh,
            'Public_Charged_kWh': public_charged_kwh,
            'Public_Arrival_Time_h': public_arrival_hour,
            'SoC_at_Home_Arrival_kWh': current_soc_kwh,
            'Energy_Gap_Home_kWh': energy_gap_kwh,
            'Home_Charging_Power_kW': home_charger_power,
            'Home_Charging_Schedule_kW': home_charging_schedule.tolist(),
            'Total_Home_Energy_Charged_kWh': sum(home_charging_schedule)
        }
        ev_records.append(record)
        total_hourly_demand += home_charging_schedule

    return pd.DataFrame(ev_records), total_hourly_demand, total_hourly_public_demand

# --- Run Simulations ---

# Run unoptimized simulation
ev_df_unoptimized, total_hourly_home_demand_unoptimized, total_hourly_public_demand_unoptimized = \
    simulate_charging(base_ev_params, simulation_type='unoptimized')

# Run smart simulation (public demand will be identical if using same base_ev_params)
ev_df_smart, total_hourly_home_demand_smart, total_hourly_public_demand_smart = \
    simulate_charging(base_ev_params, simulation_type='smart')

# --- Cost Comparison ---
home_cost_unoptimized = calculate_charging_cost(total_hourly_home_demand_unoptimized, hourly_prices)
home_cost_smart = calculate_charging_cost(total_hourly_home_demand_smart, hourly_prices)

# Calculate public charging cost. This is total kWh from public charging * avg public price.
# Since public_charged_kwh is in the dataframe, sum it up.
total_public_charged_kwh = ev_df_unoptimized['Public_Charged_kWh'].sum() # It's the same for both scenarios
total_public_charging_cost = total_public_charged_kwh * public_avg_price_per_kwh

total_fleet_cost_unoptimized = home_cost_unoptimized + total_public_charging_cost
total_fleet_cost_smart = home_cost_smart + total_public_charging_cost

cost_saving_absolute = total_fleet_cost_unoptimized - total_fleet_cost_smart
cost_saving_percentage = (cost_saving_absolute / total_fleet_cost_unoptimized) * 100 if total_fleet_cost_unoptimized > 0 else 0

# Cost comparison 
print("\n--- Cost Comparison (Daily for the entire fleet) ---")
print(f"Total Home Charging Cost (Unoptimized): ${home_cost_unoptimized:.2f}")
print(f"Total Home Charging Cost (Smart Optimized): ${home_cost_smart:.2f}")
print(f"Total Public Charging Cost (Fleet): ${total_public_charging_cost:.2f}")
print(f"--------------------------------------------------")
print(f"Total Fleet Charging Cost (Unoptimized Home + Public): ${total_fleet_cost_unoptimized:.2f}")
print(f"Total Fleet Charging Cost (Smart Home + Public): ${total_fleet_cost_smart:.2f}")
print(f"Cost Savings from Smart Charging: ${cost_saving_absolute:.2f} ({cost_saving_percentage:.2f}%)")

# --- Analysis and Visualization ---
print("--- EV Fleet Charging Summary (Unoptimized) ---")
print(f"Total Home Energy Charged (Unoptimized): {ev_df_unoptimized['Total_Home_Energy_Charged_kWh'].sum():.2f} kWh")
print(f"Peak Demand (Unoptimized): {total_hourly_home_demand_unoptimized.max():.2f} kW")

print("\n--- EV Fleet Charging Summary (Smart Optimized) ---")
print(f"Total Home Energy Charged (Smart): {ev_df_smart['Total_Home_Energy_Charged_kWh'].sum():.2f} kWh")
print(f"Peak Demand (Smart): {total_hourly_home_demand_smart.max():.2f} kW")

print(f"\nTotal Public Energy Charged (Fleet): {total_hourly_public_demand_unoptimized.sum():.2f} kWh (should be same for both)")


# Plotting the aggregated demand profiles
plt.figure(figsize=(14, 8))
plt.plot(np.arange(24), total_hourly_home_demand_smart, label='Aggregated Home Charging Demand (Smart)')
plt.plot(np.arange(24), total_hourly_home_demand_unoptimized, label='Aggregated Home Charging Demand (Unoptimized)', linestyle='--')
plt.plot(np.arange(24), total_hourly_public_demand_unoptimized, label='Aggregated Public Charging Demand', color='red') # Can use either unoptimized or smart version, they are the same

plt.plot(np.arange(24), total_hourly_home_demand_smart + total_hourly_public_demand_unoptimized,
         label='Total Fleet Demand (Smart Home + Public)', linestyle='-.', color='green')
plt.plot(np.arange(24), total_hourly_home_demand_unoptimized + total_hourly_public_demand_unoptimized,
         label='Total Fleet Demand (Unoptimized Home + Public)', linestyle=':', color='purple')
plt.legend(loc='upper center', fontsize=9)

# Plot prices on secondary axis
ax2 = plt.gca().twinx()
ax2.plot(np.arange(24), hourly_prices * 100, label='Electricity Price (cents/kWh)', color='gray', linestyle=':')
ax2.set_ylabel('Price (cents/kWh)', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

plt.xticks(np.arange(0, 24))
plt.xlabel('Hour of Day')
plt.ylabel('Power (kW)')
plt.title('Aggregated EV Charging Demand Profiles: Smart vs. Unoptimized Home Charging')
plt.grid(True)
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Plotting the GMM distribution for public arrival times
plt.figure(figsize=(10, 6))
x_vals = np.linspace(0, 24, 500)
gmm_pdf_values = np.zeros_like(x_vals)
for i in range(len(active_gmm_weights)):
    gmm_pdf_values += active_gmm_weights[i] * (1 / (active_gmm_sigmas[i] * np.sqrt(2 * np.pi))) * \
                      np.exp(-((x_vals - active_gmm_mus[i])**2 / (2 * active_gmm_sigmas[i]**2)))

plt.plot(x_vals, gmm_pdf_values, label='GMM Probability Density Function')
sns.histplot(ev_df_unoptimized['Public_Arrival_Time_h'].dropna(), bins=24, stat='density', color='orange', alpha=0.6, label='Simulated Public Arrival Times')
plt.title('Public Charging Arrival Time Distribution (GMM)')
plt.xlabel('Hour of Day')
plt.ylabel('Probability Density')
plt.xticks(np.arange(0, 25, 2))
plt.grid(True)
plt.legend()
plt.show()

# Additional plots for individual parameter distributions (using unoptimized dataframe as base)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(ev_df_unoptimized['Daily_Distance_km'], bins=30, kde=True, label='Daily Travel Distance')
plt.title('Distribution of Daily Travel Distance')
plt.xlabel('Distance (km)')
plt.legend()

plt.subplot(2, 3, 2)
sns.histplot(ev_df_unoptimized['Energy_Gap_Home_kWh'], bins=30, kde=True, label='Home Charging Energy Gap')
plt.title('Distribution of Home Charging Energy Gap')
plt.xlabel('Energy (kWh)')
plt.legend()

plt.subplot(2, 3, 3)
sns.histplot(ev_df_unoptimized['Home_Charging_Power_kW'], bins=len(home_charging_powers), kde=False, discrete=True, label='Home Charging Power')
plt.title('Distribution of Home Assigned Charging Power')
plt.xlabel('Power (kW)')
plt.xticks(home_charging_powers)
plt.legend()

plt.subplot(2, 3, 4)
sns.histplot(ev_df_unoptimized['Home_Arrival_Time_h'], bins=24, kde=True, label='Home Arrival Time')
plt.title('Distribution of Home Arrival Time')
plt.xlabel('Time (Hours)')
plt.xticks(range(0, 25, 2))
plt.legend()

plt.subplot(2, 3, 5)
ev_charged_home_unoptimized = ev_df_unoptimized[ev_df_unoptimized['Total_Home_Energy_Charged_kWh'] > 0.1]
if not ev_charged_home_unoptimized.empty:
    rough_duration_unoptimized = ev_charged_home_unoptimized['Total_Home_Energy_Charged_kWh'] / ev_charged_home_unoptimized['Home_Charging_Power_kW']
    sns.histplot(rough_duration_unoptimized, bins=30, kde=True, label='Home Charging Duration (Unoptimized)')
    plt.title('Distribution of Home Charging Duration (Unoptimized)')
    plt.xlabel('Duration (Hours)')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No unoptimized home charging data to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


plt.tight_layout()
plt.show()

# If you specifically want to output the unoptimized data frame for separate use:
#ev_df_unoptimized.to_csv('ev_demand_unoptimized.csv', index=False)
#print("\nUnoptimized EV demand data saved to ev_demand_unoptimized.csv")